# coding: utf-8
import os
from src.config import args
os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
from src.train_and_evaluate import *
from src.models import *
from src.util import *
import time
import torch.optim
from src.expressions_transfer import *
import transformers
from collections import defaultdict

batch_size = 5
hidden_size = 512
n_epochs = 85
learning_rate = 3e-4
bert_lr = 3e-5
weight_decay = 1e-5
beam_size = 5
n_layers = 2
embedding_size = 128
dropout = 0.5
model_file = "models"
if not os.path.isdir(model_file):
    os.makedirs(model_file)

BertTokenizer_path = "bert_model/bert-base-chinese" 
Bert_model_path = "bert_model/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(BertTokenizer_path)
tokenizer.add_special_tokens({"additional_special_tokens":["[NUM]"]})

best_acc = []
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename='log')

def get_formula_fold(data,pairs):
    new_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair.append(item['formula'])
        # pair.append(item['id'])
        pair = tuple(pair)
        new_fold.append(pair)
    return new_fold

def load_formulas(file):
    with open(file,'r') as f:
        formula = f.read()
    formulas = formula.split('\n')
    formulas = [i.strip(' ') for i in formulas]
    formula_ent = set(formula.replace('\n',' ').split(' '))- set("+-*/^()=")-{''}
    return formulas, formula_ent

all_formulas, all_formula_ent = load_formulas('data/Math23K/formula.txt')
## all_formula_exp
all_formulas_exp, _ = load_formulas('data/Math23K/formula_variant.txt')

formula_exp_dict = dict(zip(all_formulas_exp, range(len(all_formulas_exp))))
formula_exp_dict['None'] = len(formula_exp_dict)
formula_ent_dict = dict(zip(all_formula_ent, range(len(all_formula_ent))))

train_data_path = "./data/Math23K/train23k_processed.json"
test_data_path = "./data/Math23K/test23k_processed.json"
train_data = load_raw_data1(train_data_path)
test_data = load_raw_data1(test_data_path)

# 过滤高频词
word_count = defaultdict(int)
for d in train_data:
    for w in d['new_split'].split(' '):
        word_count[w] += 1
filtered_words = []
for w in word_count:
    if word_count[w]/len(train_data) < 0.1:
        filtered_words.append(w)

filtered_wo_words = []
for w in word_count:
    if word_count[w]/len(train_data) < 0.01:
        filtered_wo_words.append(w)
    
# ConceptNet
concept_know = np.load("./data/Math23K/math23k_know.npy",'r')
ww_base = defaultdict(list)
form_of = defaultdict(list)
for d in concept_know:  # 过滤formof关系的知识
    x,y,z = d
    if (z == 'FormOf') and (x != y):
        if y not in form_of[x]:
            form_of[x].append(y)
        if x not in form_of[y]:
            form_of[y].append(x)
for d in concept_know:
    x,y,z = d
    x_token, y_token = tokenizer.tokenize(x), tokenizer.tokenize(y)
    if (len(x_token)==1) and (len(y_token)==1) and (x in filtered_words) and (y in filtered_words):
        if (z == 'RelatedTo') and (x != y):
            if (y_token[0] not in ww_base[x_token[0]]) and (y not in form_of[x]):
                ww_base[x_token[0]].append(y_token[0])
            if (x_token[0] not in ww_base[y_token[0]]) and (x not in form_of[y]):
                ww_base[y_token[0]].append(x_token[0])

pairs1, generate_nums1, copy_nums1 = transfer_num1(train_data,tokenizer=tokenizer)
pairs2, _ , _ = transfer_num1(test_data,tokenizer=tokenizer)

temp_pairs1 = []
for p in pairs1:
    temp_pairs1.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3], p[4], p[5]))
pairs1 = temp_pairs1

temp_pairs2 = []
for p in pairs2:
    temp_pairs2.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3], p[4], p[5]))
pairs2 = temp_pairs2

pairs_trained = get_formula_fold(train_data, pairs1)
pairs_tested = get_formula_fold(test_data, pairs2)
input_lang, output_lang, train_pairs, test_pairs = prepare_data1(pairs_trained, pairs_tested, 5, generate_nums1,copy_nums1, tokenizer=tokenizer,formula_exp_dict=formula_exp_dict, tree=True)
question_list = [item["segmented_text"] for item in test_data]
# ww_list = [item["ww_know"] for item in test_data]
# wo_list = [item["wo_know"] for item in test_data]

del train_data,test_data,pairs1,pairs2,temp_pairs1,temp_pairs2,pairs_trained,pairs_tested

# Initialize models
encoder = EncoderBert(hidden_size=hidden_size,bert_pretrain_path=Bert_model_path,dropout=dropout)
# encoder.load_t5(t5.state_dict())
encoder2 = EncoderBert(hidden_size=hidden_size,bert_pretrain_path=Bert_model_path,dropout=dropout)
# encoder2.load_t5(t5.state_dict())
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            input_size=len(generate_nums1))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
formula_enc = Formula_Encoding(formula_exp_dict=formula_exp_dict, formula_ent_dict=formula_ent_dict, hidden_size=hidden_size, embedding_size=embedding_size, word2index=output_lang.word2index)
verify = Verify(hidden_size=hidden_size)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=bert_lr, weight_decay=weight_decay)
encoder2_optimizer = torch.optim.Adam(encoder2.parameters(), lr=bert_lr, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
verify_optimizer = torch.optim.Adam(verify.parameters(), lr=0.001, weight_decay=weight_decay)
formula_pretrain_optimizer = torch.optim.Adam(formula_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)
formula_enc_optimizer = torch.optim.Adam(formula_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=30, gamma=0.5)
encoder2_scheduler = torch.optim.lr_scheduler.StepLR(encoder2_optimizer, step_size=30, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=30, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=30, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=30, gamma=0.5)
verify_scheduler = torch.optim.lr_scheduler.StepLR(verify_optimizer, step_size=30, gamma=0.5)
formula_pretrain_scheduler = torch.optim.lr_scheduler.StepLR(formula_pretrain_optimizer, step_size=30, gamma=0.5)
formula_enc_scheduler = torch.optim.lr_scheduler.StepLR(formula_enc_optimizer, step_size=30, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    encoder2.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    verify.cuda()
    formula_enc.cuda()

generate_num_ids = []
for num in generate_nums1:
    generate_num_ids.append(output_lang.word2index[num])
best_val_cor = 0
best_eval_total  = 1

best_formula_acc = 0
# formula understanding pretaining
for epoch in range(100):
    formula_pretrain_scheduler.step()
    loss_pretrain = 0
    loss = pretrain_formula(formula_enc, formula_pretrain_optimizer)
    loss_pretrain += loss
    logging.info(f"loss_pretrain: {loss_pretrain}")
        
    formula_acc = eval_formula(formula_enc)
    logging.info(f"formula acc: {formula_acc}")
    if formula_acc > best_formula_acc:
        best_formula_acc = formula_acc
        torch.save(formula_enc.state_dict(), "models/formula_enc")
formula_enc.load_state_dict(torch.load("models/formula_enc"))

accum_iter = 2
for epoch in range(n_epochs):
    encoder_scheduler.step()
    encoder2_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    formula_enc_scheduler.step()
    verify_scheduler.step()
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, step_batches, step_lengths, step_label_batches, formula_batches = prepare_train_batch1(train_pairs, batch_size)
    logging.info(f"epoch: {epoch + 1}")
    start = time.time()
    for idx in range(len(input_lengths)):
        loss = train_verifier(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], step_batches[idx], step_lengths[idx], step_label_batches[idx], generate_num_ids, encoder, encoder2, predict, generate, merge, formula_enc, verify, 
            encoder_optimizer, encoder2_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, formula_enc_optimizer, verify_optimizer, input_lang, output_lang, num_pos_batches[idx], formula_batches[idx])
        if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(input_lengths)):
            # Update parameters with optimizers
            encoder_optimizer.step()
            encoder2_optimizer.step()
            predict_optimizer.step()
            generate_optimizer.step()
            merge_optimizer.step()
            verify_optimizer.step()
            formula_enc_optimizer.step()

            # Zero gradients of both optimizers
            encoder_optimizer.zero_grad()
            encoder2_optimizer.zero_grad()
            predict_optimizer.zero_grad()
            generate_optimizer.zero_grad()
            merge_optimizer.zero_grad()
            formula_enc_optimizer.zero_grad()
            verify_optimizer.zero_grad()
        loss_total += loss

    logging.info(f"loss: {loss_total / len(input_lengths)}")
    logging.info(f"training time: {time_since(time.time() - start)}")
    # print("--------------------------------")
    #开始valid
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    start = time.time()
    #pairs:input_seq, len(input_seq), out_seq(prefix with index), len(out_seq), nums, num_pos, num_stack
    for idx, test_batch in enumerate(test_pairs):
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, encoder2, predict, generate,
                                        merge, formula_enc, verify, input_lang,output_lang, test_batch[5], beam_size=beam_size)
        val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1
        
    logging.info(f"valid_eq_acc: {float(equation_ac) / eval_total}, valid_an_acc: {float(value_ac) / eval_total}")
    # verifier验证
    acc_cons, acc_alls, pre_rec_cons, pre_alls, rec_alls = 0, 0, 0, 0, 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, step_batches, step_lengths, step_label_batches, formula_batches = prepare_train_batch1(test_pairs, batch_size)
    for idx in range(len(input_lengths)):
        acc_con, acc_all, pre_rec_con, pre_all, rec_all = evaluate_verifier(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], step_batches[idx], step_lengths[idx], step_label_batches[idx], generate_num_ids, encoder, encoder2, predict, generate, merge, formula_enc, verify, input_lang,output_lang, num_pos_batches[idx])
        acc_cons += acc_con
        acc_alls += acc_all
        pre_rec_cons += pre_rec_con
        pre_alls += pre_all
        rec_alls += rec_all

    logging.info(f"valid_verify_acc: {float(acc_cons) / acc_alls}, valid_verify_pre: {float(pre_rec_cons) / pre_alls}, valid_verify_rec: {float(pre_rec_cons) / rec_alls}")
    logging.info(f"time: {time_since(time.time() - start)}")
    # print("------------------------------------------------------")
    if float(value_ac) / eval_total > best_val_cor / best_eval_total:
        best_val_cor = value_ac
        best_eval_total = eval_total
        torch.save(encoder.state_dict(), "models/encoder")
        torch.save(encoder2.state_dict(), "models/encoder2")
        torch.save(predict.state_dict(), "models/predict")
        torch.save(generate.state_dict(), "models/generate")
        torch.save(merge.state_dict(), "models/merge")
        torch.save(formula_enc.state_dict(), "models/formula_enc")
        torch.save(verify.state_dict(), "models/verify")
    
encoder.load_state_dict(torch.load("models/encoder"))
encoder2.load_state_dict(torch.load("models/encoder2"))
predict.load_state_dict(torch.load("models/predict"))
generate.load_state_dict(torch.load("models/generate"))
merge.load_state_dict(torch.load("models/merge"))
formula_enc.load_state_dict(torch.load("models/formula_enc"))
verify.load_state_dict(torch.load("models/verify"))
value_ac = 0
equation_ac = 0
eval_total = 0
for idx, test_batch in enumerate(test_pairs):
    test_res = evaluate_tree_GPT3(test_batch[0], test_batch[1], generate_num_ids, encoder, encoder2, predict, generate,
                                    merge, formula_enc, verify, input_lang,output_lang, test_batch[5], nums_list=test_batch[4], question=question_list[idx], ww_know=ww_list[idx], wo_know=wo_list[idx], formulas=all_formulas_exp, tokenizer=tokenizer, beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
logging.info(f"Final valid_eq_acc: {float(equation_ac) / eval_total}, valid_an_acc: {float(value_ac) / eval_total}")
