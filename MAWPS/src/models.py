# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
from torch.nn.parameter import Parameter
from transformers import BertModel, BertTokenizer


class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size, dropout=0.5):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)

    def forward(self, input_seqs):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        return embedded


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:,
                                                             :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs),
                              2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(
            self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(
            max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size,
                          hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(
            1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(
            last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(
            torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(
            torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class FormulaNode:  # the class save the formula node
    def __init__(self, embedding, symbol_embedding, left_child, right_child, symbol_index=None):
        self.embedding = embedding
        self.symbol_embedding = symbol_embedding
        self.left_child = left_child
        self.right_child = right_child
        self.symbol_index = symbol_index


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings),
                              2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs),
                              2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(
            max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class FormulaAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FormulaAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, formula_score, encoder_outputs, seq_mask=None):
        # formula_score: r x dim
        # encoder_outputs: s x b x dim
        max_len, this_batch_size, d = encoder_outputs.size()
        r_num = formula_score.size(0)
        
        encoder_outputs_1 = encoder_outputs.contiguous().view(-1, d).unsqueeze(0)  # 1 x (sb) x dim
        repeat_dims = [1] * encoder_outputs_1.dim()
        repeat_dims[0] = r_num
        encoder_outputs_1 = encoder_outputs_1.repeat(*repeat_dims)  # r x (SB) x dim

        formula_1 = formula_score.unsqueeze(1).repeat([1, max_len*this_batch_size, 1])  # r x (SB) x dim
        energy_in = torch.cat((encoder_outputs_1, formula_1),
                              2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (rSB) x 1
        attn_energies = attn_energies.squeeze(-1)
        attn_energies = attn_energies.view(
            r_num, max_len, this_batch_size).transpose(1, 2).transpose(0, 1)  # B x r x S
        if seq_mask is not None:
            formula_seq_mask = seq_mask.unsqueeze(1).repeat([1, r_num, 1])
            attn_energies = attn_energies.masked_fill_(formula_seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=-1)  # B x r x S

        return attn_energies
    
    
class BertEncoder(nn.Module):
    def __init__(self, bert_model, dropout=0.5):
        super(BertEncoder, self).__init__()
        self.bert_layer = BertModel.from_pretrained(bert_model)
        self.em_dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        output = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embedded = self.em_dropout(output[0]) # B x S x Bert_emb(768)
        return embedded

class EncoderSeq(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru_pade = nn.LSTM(embedding_size, hidden_size,
                               n_layers, dropout=dropout, bidirectional=True)

    def forward(self, embedded, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + \
            pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + \
            pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output

class EncoderBert(nn.Module):
    def __init__(self, hidden_size, bert_pretrain_path='', dropout=0.5):
        super(EncoderBert, self).__init__()
        self.embedding_size = 768
        print("bert_model: ", bert_pretrain_path)
        self.bert_model = BertModel.from_pretrained(bert_pretrain_path)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.em_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embedded = self.em_dropout(output[0]) # B x S x Bert_emb(768)
        
        pade_outputs = self.linear(embedded) # B x S x E
        pade_outputs = pade_outputs.transpose(0,1) # S x B x E

        problem_output = pade_outputs[0]
        return pade_outputs, problem_output

class Verify(nn.Module):
    def __init__(self, hidden_size):
        super(Verify, self).__init__()
        
        self.hidden_size = hidden_size
        self.verify1 = nn.Linear(hidden_size * 2, hidden_size)
        self.verify2 = nn.Linear(hidden_size, 2)
        
    def forward(self, current_node, step_output):
        verify_score = torch.tanh(self.verify1(torch.cat([current_node, step_output], dim=-1)))
        verify_prob = torch.softmax(self.verify2(verify_score).squeeze(-1), dim=-1)
        return verify_prob
   
class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(
            torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.verse_attn = TreeAttn(hidden_size, hidden_size)
        self.formula_attn = FormulaAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

        self.formula_probs1 = nn.Linear(hidden_size, hidden_size)
        self.formula_probs2 = nn.Linear(hidden_size, 1)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums, formula_scores, formula_roots, formula_none_id, verify, step_output=None, step_outputs=None, step_label=None, step_mask=None, train=False, formula_gt=None):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)
        
        step_info = 0
        verse_step_info = 0
        if step_output != None:
        # verifiy
            verify_prob = verify(current_node.squeeze(1), step_output)
            step_attn = self.attn(current_node.transpose(0, 1), step_outputs, step_mask)
            # print(step_outputs.size(), encoder_outputs.size(), current_node.size(), step_output.size())
            verse_step_attn = self.verse_attn(step_output.unsqueeze(0), encoder_outputs, seq_mask)

            step_context = step_attn.bmm(step_outputs.transpose(0, 1)).squeeze(1)  # B x N
            verse_step_context = verse_step_attn.bmm(encoder_outputs.transpose(0, 1)).squeeze(1)  # B x N
            if step_label != None:  # train mode
                step_info = (step_context * step_label).unsqueeze(1)
                verse_step_info = (verse_step_context * step_label).unsqueeze(1)
            elif verify_prob[0][1] > verify_prob[0][0]:  # test mode
                step_info = step_context.unsqueeze(1)
                verse_step_info = verse_step_context.unsqueeze(1)
        else:
            verify_prob = None
        # print(current_node.size(), step_info.size())
        current_node = current_node + 0.5 * step_info + 0.5 * verse_step_info
        current_embeddings = self.dropout(current_node) 

        current_attn = self.attn(current_embeddings.transpose(
            0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(
            encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(
            *repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat(
            (embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # formula attn
        formulas_attn = self.formula_attn(formula_scores, encoder_outputs, seq_mask)
        formulas_context = (formulas_attn+0.2*current_attn).bmm(
            encoder_outputs.transpose(0, 1))  # B x r x N

        #formula selection
        formula_score = torch.tanh(self.formula_probs1(formulas_context))
        formula_prob = torch.softmax(self.formula_probs2(formula_score).squeeze(-1), dim=-1)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1),
                               embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight, formula_prob, verify_prob


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(
            torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(
            torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(
            torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(
            torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(
            torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(
            torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class Formula_Judgement(nn.Module):
    def __init__(self, hidden_size):
        super(Formula_Judgement, self).__init__()
        self.f1 = nn.Linear(hidden_size, int(hidden_size))
        self.f2 = nn.Linear(int(hidden_size), int(hidden_size/2))
        self.output = nn.Linear(int(hidden_size/2), 1)
    
    def forward(self, formula_scores):
        # formula_scores: n x hidden_size
        o1 = torch.tanh(self.f1(formula_scores))
        out = torch.sigmoid(self.output(torch.relu(self.f2(o1))))
        return out
        
class Formula_Encoding(nn.Module):
    def __init__(self, formula_exp_dict, formula_ent_dict, hidden_size, embedding_size, word2index, dropout=0.5):
        super(Formula_Encoding, self).__init__()
        
        self.formula_exp_dict = formula_exp_dict
        self.formula_ent_dict = formula_ent_dict
        self.ent_num = len(formula_ent_dict)
        self.ent_emb = nn.Embedding(self.ent_num, hidden_size)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.op = ['+','-','*','/','=']
        self.op_emb = nn.Embedding(len(self.op), hidden_size)
        self.merge = Merge(hidden_size, hidden_size)
        self.output_word2index = word2index
        
        self.formula_judge = Formula_Judgement(hidden_size)
        self.non_embed = nn.Embedding(1, hidden_size)
        self.init_encoding()
    
    def init_encoding(self):
        all_ids = []
        all_words = []
        for formula in self.formula_exp_dict:
            if formula != "None":
                ids, words = self.id_convert(formula)
                all_ids.append(ids)
                all_words.append(words)
        self.all_ids = all_ids
        self.all_words = all_words
    
    def id_convert(self, formula):
        formula_split = formula.split(' ')
        formula_id = []
        for i in formula_split:
            if i in self.op:
                formula_id.append([self.op.index(i)])
            else:
                formula_id.append([self.formula_ent_dict[i]])
        return torch.LongTensor(formula_id), formula_split
       
    def forward(self, cuda=True):
        all_score = []
        all_root = []
        for (formula_ids, formula_words) in zip(self.all_ids, self.all_words):
            if cuda:
                formula_ids = formula_ids.cuda()
            score, root = self.single_encoding(formula_ids, formula_words)
            all_score.append(score)
            all_root.append(root)
        non_embed = torch.LongTensor([0])
        if cuda:
            non_embed = non_embed.cuda()
        all_score.append(self.non_embed(non_embed))
        all_score = torch.cat(all_score, dim=0)
        all_root.append(None)
        return all_score, all_root
        
    def single_encoding(self, formula_ids, formula_words):
        left, eq = formula_ids[0], formula_ids[1]
        formula_l = formula_ids[2:]
        node_stack, formula_chain = [], []
        for i in range(len(formula_l)):
            w, si = formula_words[i+2], formula_ids[i+2]
            if w in ['+','-','*','/']:
                node_stack.append(TreeEmbedding(self.op_emb(si), False))
                formula_chain.append(FormulaNode(None, self.op_emb(si), None, None, self.output_word2index[w]))
            elif not node_stack[-1].terminal:
                node_stack.append(TreeEmbedding(self.ent_emb(si), True))
                formula_chain.append(FormulaNode(self.ent_emb(si), self.ent_emb(si), None, None))
            else:
                left_num = node_stack.pop()
                right_num = self.ent_emb(si)
                op = node_stack.pop()
                new_num = self.merge(op.embedding, left_num.embedding, right_num)
                node_stack.append(TreeEmbedding(new_num, True))
                
                left_formula = formula_chain.pop()
                op_formula = formula_chain.pop()
                op_formula.embedding = new_num
                op_formula.left_child = left_formula
                op_formula.right_child = FormulaNode(right_num, right_num, None, None)
                formula_chain.append(op_formula)
        while len(node_stack) > 1:
            right_num = node_stack.pop()
            left_num = node_stack.pop()
            op = node_stack.pop()
            new_num = self.merge(op.embedding, left_num.embedding, right_num.embedding)
            node_stack.append(TreeEmbedding(new_num, True))
            
            right_formula = formula_chain.pop()
            left_formula = formula_chain.pop()
            op_formula = formula_chain.pop()
            op_formula.embedding = new_num
            op_formula.left_child = left_formula
            op_formula.right_child = right_formula
            formula_chain.append(op_formula)
        root_emb = self.op_emb(eq)
        left_emb = self.ent_emb(left)
        score = self.merge(root_emb, left_emb, node_stack[0].embedding)
        chain_root = FormulaNode(score, root_emb, FormulaNode(left_emb, left_emb, None, None), formula_chain[0])
        return score, chain_root
    
    def generate_false_formula(self, cuda=True):
        false_score = []
        for (formula_ids, formula_words) in zip(self.all_ids, self.all_words):
            if cuda:
                formula_ids = formula_ids.cuda()
            formula_ids1 = copy.deepcopy(formula_ids)
            formula_words1 = copy.deepcopy(formula_words)
            pos_ = random.sample(range(len(formula_ids1)-2),1)[0] + 2
            if formula_words[pos_] in self.op:
                op_can = ['+','-','*','/']
                op_can.remove(formula_words[pos_])
                neg_word = random.sample(op_can, 1)[0]
                formula_ids1[pos_] = self.op.index(neg_word)
            else:
                word_can = list(self.formula_ent_dict)
                word_can.remove(formula_words[pos_])
                neg_word = random.sample(word_can, 1)[0]
                formula_ids1[pos_] = self.formula_ent_dict[neg_word]
            formula_words1[pos_] = neg_word
            score, _ = self.single_encoding(formula_ids1, formula_words1)
            false_score.append(score)
        false_prob = self.formula_judge(torch.cat(false_score))
        return false_prob
            
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
        
class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
    
# Graph_Conv
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.shape)
        #print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        #print(adj.shape)
        #print(support.shape)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'