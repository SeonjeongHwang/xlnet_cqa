from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('xlnet') # walkaround due to submodule absolute import...

import collections
import os
import os.path
import json
import pickle
import time
import string

import tensorflow as tf
import numpy as np
import sentencepiece as sp
from six.moves import xrange

from tool.eval_coqa import CoQAEvaluator
from xlnet import xlnet
import function_builder
import prepro_utils
import model_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MAX_FLOAT = 1e30
MIN_FLOAT = -1e30

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "Data directory where raw data located.")
flags.DEFINE_string("output_dir", None, "Output directory where processed data located.")
flags.DEFINE_string("model_dir", None, "Model directory where checkpoints located.")
flags.DEFINE_string("export_dir", None, "Export directory where saved model located.")

flags.DEFINE_string("task_name", default=None, help="The name of the task to train.")
flags.DEFINE_string("model_config_path", default=None, help="Config file of the pre-trained model.")
flags.DEFINE_string("init_checkpoint", default=None, help="Initial checkpoint of the pre-trained model.")
flags.DEFINE_string("spiece_model_file", default=None, help="Sentence Piece model path.")
flags.DEFINE_bool("overwrite_data", default=False, help="If False, will use cached data if available.")
flags.DEFINE_string("predict_tag", None, "Predict tag for predict result tracking.")

flags.DEFINE_bool("do_train", default=False, help="Whether to run training.")
flags.DEFINE_bool("do_predict", default=False, help="Whether to run prediction.")
flags.DEFINE_bool("do_export", default=False, help="Whether to run exporting.")

flags.DEFINE_integer("random_seed", default=100, help="Random seed for weight initialzation.")
flags.DEFINE_enum("init", default="normal", enum_values=["normal", "uniform"], help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1, help="Initialization std when init is uniform.")
flags.DEFINE_bool("init_global_vars", default=False, help="If true, init all global vars. If false, init trainable vars only.")

flags.DEFINE_bool("lower_case", default=False, help="Enable lower case nor not.")
flags.DEFINE_integer("num_turn", default=2, help="Number of turns.")
flags.DEFINE_integer("doc_stride", default=128, help="Doc stride")
flags.DEFINE_integer("max_seq_length", default=512, help="Max sequence length")
flags.DEFINE_integer("max_query_length", default=128, help="Max query length")
flags.DEFINE_integer("train_batch_size", default=48, help="Total batch size for training.")
flags.DEFINE_integer("predict_batch_size", default=32, help="Total batch size for predict.")

flags.DEFINE_integer("train_steps", default=20000, help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_integer("max_save", default=5, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=1000, help="Save the model for every save_steps. If None, not to save any model.")
flags.DEFINE_integer("shuffle_buffer", default=2048, help="Buffer size used for shuffle.")

flags.DEFINE_string("target_eval_key", default="best_f1", help="Use has_ans_f1 for Model I.")

flags.DEFINE_bool("use_bfloat16", default=False, help="Whether to use bfloat16.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")
flags.DEFINE_string("summary_type", default="last", help="Method used to summarize a sequence into a vector.")

flags.DEFINE_float("learning_rate", default=3e-5, help="initial learning rate")
flags.DEFINE_float("min_lr_ratio", default=0.0, help="min lr ratio for cos decay.")
flags.DEFINE_float("lr_layer_decay_rate", default=0.75, help="lr[L] = learning_rate, lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("num_hosts", 1, "How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", 1, "Total number of TPU cores to use.")
flags.DEFINE_string("tpu_job_name", None, "TPU worker job name.")
flags.DEFINE_string("tpu", None, "The Cloud TPU name to use for training.")
flags.DEFINE_string("tpu_zone", None, "GCE zone where the Cloud TPU is located in.")
flags.DEFINE_string("gcp_project", None, "Project name for the Cloud TPU-enabled project.")
flags.DEFINE_string("master", None, "TensorFlow master URL")
flags.DEFINE_integer("iterations", 1000, "number of iterations per TPU training loop.")


class InputExample(object):
    """A single CoQA example."""
    def __init__(self,
                 para_text,
                 examples):
        self.para_text = para_text
        self.examples = examples
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = "paragraph_text: %s" % (prepro_utils.printable_text(self.para_text))
        return "[{0}]\n".format(s)
    
class eachExample(object):
    """A single CoQA example."""
    def __init__(self,
                 qas_id,
                 unique_id,
                 question_text,
                 rationale_text,
                 answer_span_text,
                 orig_answer_text,
                 label):
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.question_text = question_text
        self.rationale_text = rationale_text
        self.answer_span_text = answer_span_text
        self.orig_answer_text = orig_answer_text
        self.label = label

class InputFeatures(object):
    """A single CoQA feature."""
    def __init__(self,
                 qas_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label):
        self.qas_id = qas_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

class OutputResult(object):
    """A single CoQA result."""
    def __init__(self,
                 qas_id,
                 prob):
        self.qas_id = qas_id
        self.prob = prob
        
class CoqaPipeline(object):
    """Pipeline for CoQA dataset."""
    def __init__(self,
                 data_dir,
                 task_name,
                 num_turn):
        self.data_dir = data_dir
        self.task_name = task_name
        self.num_turn = num_turn
    
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        data_path = os.path.join(self.data_dir, "train-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "dev-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def _read_json(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)["data"]
                return data_list
        else:
            raise FileNotFoundError("data path not found: {0}".format(data_path))
    
    def _whitespace_tokenize(self,
                             text):
        word_spans = []
        char_list = []
        for idx, char in enumerate(text):
            if char != ' ':
                char_list.append(idx)
                continue
            
            if char_list:
                word_start = char_list[0]
                word_end = char_list[-1]
                word_text = text[word_start:word_end+1]
                word_spans.append((word_text, word_start, word_end))
                char_list.clear()
        
        if char_list:
            word_start = char_list[0]
            word_end = char_list[-1]
            word_text = text[word_start:word_end+1]
            word_spans.append((word_text, word_start, word_end))
        
        return word_spans
    
    def _char_span_to_word_span(self,
                                char_start,
                                char_end,
                                word_spans):
        word_idx_list = []
        for word_idx, (_, start, end) in enumerate(word_spans):
            if end >= char_start:
                if start <= char_end:
                    word_idx_list.append(word_idx)
                else:
                    break
        
        if word_idx_list:
            word_start = word_idx_list[0]
            word_end = word_idx_list[-1]
        else:
            word_start = -1
            word_end = -1
        
        return word_start, word_end
    
    def _search_best_span(self,
                          context_tokens,
                          answer_tokens):
        best_f1 = 0.0
        best_start, best_end = -1, -1
        search_index = [idx for idx in range(len(context_tokens)) if context_tokens[idx][0] in answer_tokens]
        for i in range(len(search_index)):
            for j in range(i, len(search_index)):
                candidate_tokens = [context_tokens[k][0] for k in range(search_index[i], search_index[j]+1) if context_tokens[k][0]]
                common = collections.Counter(candidate_tokens) & collections.Counter(answer_tokens)
                num_common = sum(common.values())
                if num_common > 0:
                    precision = 1.0 * num_common / len(candidate_tokens)
                    recall = 1.0 * num_common / len(answer_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_start = context_tokens[search_index[i]][1]
                        best_end = context_tokens[search_index[j]][2]
        
        return best_f1, best_start, best_end
    
    def _get_question_text(self,
                           history,
                           question):
        question_tokens = ['<s>'] + question["input_text"].split(' ')
        return " ".join(history + [" ".join(question_tokens)])
    
    def _get_question_history(self,
                              history,
                              question,
                              answer,
                              answer_type,
                              is_skipped,
                              num_turn):
        question_tokens = []
        if answer_type != "unknown":
            question_tokens.extend(['<s>'] + question["input_text"].split(' '))
            question_tokens.extend(['</s>'] + answer["input_text"].split(' '))
        
        question_text = " ".join(question_tokens)
        if question_text:
            history.append(question_text)
        
        if num_turn >= 0 and len(history) > num_turn:
            history = history[-num_turn:]
        
        return history
    
    def _find_answer_span(self,
                          answer_text,
                          rationale_text,
                          rationale_start,
                          rationale_end):
        idx = rationale_text.find(answer_text)
        answer_start = rationale_start + idx
        answer_end = answer_start + len(answer_text) - 1
        
        return answer_start, answer_end
    
    def _match_answer_span(self,
                           answer_text,
                           rationale_start,
                           rationale_end,
                           paragraph_text):
        answer_tokens = self._whitespace_tokenize(answer_text)
        answer_norm_tokens = [CoQAEvaluator.normalize_answer(token) for token, _, _ in answer_tokens]
        answer_norm_tokens = [norm_token for norm_token in answer_norm_tokens if norm_token]
        
        if not answer_norm_tokens:
            return -1, -1
        
        paragraph_tokens = self._whitespace_tokenize(paragraph_text)
        
        if not (rationale_start == -1 or rationale_end == -1):
            rationale_word_start, rationale_word_end = self._char_span_to_word_span(rationale_start, rationale_end, paragraph_tokens)
            rationale_tokens = paragraph_tokens[rationale_word_start:rationale_word_end+1]
            rationale_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in rationale_tokens]
            match_score, answer_start, answer_end = self._search_best_span(rationale_norm_tokens, answer_norm_tokens)
            
            if match_score > 0.0:
                return answer_start, answer_end
        
        paragraph_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in paragraph_tokens]
        match_score, answer_start, answer_end = self._search_best_span(paragraph_norm_tokens, answer_norm_tokens)
        
        if match_score > 0.0:
            return answer_start, answer_end
        
        return -1, -1
    
    def _get_answer_span(self,
                         answer,
                         answer_type,
                         paragraph_text):
        input_text = answer["input_text"].strip().lower()
        span_start, span_end = answer["span_start"], answer["span_end"]
        if span_start == -1 or span_end == -1:
            span_text = ""
        else:
            span_text = paragraph_text[span_start:span_end].lower()
        
        if input_text in span_text:
            span_start, span_end = self._find_answer_span(input_text, span_text, span_start, span_end)
        else:
            span_start, span_end = self._match_answer_span(input_text, span_start, span_end, paragraph_text.lower())
        
        if span_start == -1 or span_end == -1:
            answer_text = ""
            is_skipped = (answer_type == "span")
        else:
            answer_text = paragraph_text[span_start:span_end+1]
            is_skipped = False
            
        span_text = answer["span_text"]
        rat_start = answer["span_start"]
        if span_text == "unknown":
            rat_text = "unknown"
        else:
            rat_text = span_text.lstrip()
            l_white_len = len(span_text)-len(rat_text)
            if l_white_len > 0:
                rat_start += l_white_len
            rat_text = rat_text.rstrip()
        
        return answer_text, is_skipped, rat_text
    
    def _normalize_answer(self,
                          answer):
        norm_answer = CoQAEvaluator.normalize_answer(answer)
        
        if norm_answer in ["yes", "yese", "ye", "es"]:
            return "yes"
        
        if norm_answer in ["no", "no not at all", "not", "not at all", "not yet", "not really"]:
            return "no"
        
        return norm_answer
    
    def _get_answer_type(self,
                         question,
                         answer):
        norm_answer = self._normalize_answer(answer["input_text"])
        
        if norm_answer == "unknown" or "bad_turn" in answer:
            return "unknown", None
        
        if norm_answer == "yes":
            return "yes", None
        
        if norm_answer == "no":
            return "no", None
        
        if norm_answer in ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]:
            return "number", norm_answer
        
        norm_question_tokens = CoQAEvaluator.normalize_answer(question["input_text"]).split(" ")
        if "or" in norm_question_tokens:
            index = norm_question_tokens.index("or")
            if index-1 >= 0 and index+1 < len(norm_question_tokens):
                if norm_answer == norm_question_tokens[index-1]:
                    norm_answer_detail = norm_answer
                    norm_answer = "option_a"
                elif norm_answer == norm_question_tokens[index+1]:
                    norm_answer_detail = norm_answer
                    norm_answer = "option_b"
        
        if norm_answer in ["option_a", "option_b"]:
            return "option", norm_answer_detail
        
        return "span", None
    
    def _process_found_answer(self,
                              raw_answer,
                              found_answer):
        raw_answer_tokens = raw_answer.split(' ')
        found_answer_tokens = found_answer.split(' ')
        
        raw_answer_last_token = raw_answer_tokens[-1].lower()
        found_answer_last_token = found_answer_tokens[-1].lower()
        
        if (raw_answer_last_token != found_answer_last_token and
            raw_answer_last_token == found_answer_last_token.rstrip(string.punctuation)):
            found_answer_tokens[-1] = found_answer_tokens[-1].rstrip(string.punctuation)
        
        return ' '.join(found_answer_tokens)
    
    def _get_example(self,
                     data_list):
        input_examples = []
        unique_id = 0
        for data in data_list:
            data_id = data["id"]
            paragraph_text = data["story"]
            each_examples = []
            
            questions = sorted(data["questions"], key=lambda x: x["turn_id"])
            answers = sorted(data["answers"], key=lambda x: x["turn_id"])
            
            question_history = []
            qas = list(zip(questions, answers))
            for i, (question, answer) in enumerate(qas):
                qas_id = "{0}_{1}".format(data_id, i+1)
                
                answer_type, answer_subtype = self._get_answer_type(question, answer)
                answer_text, is_skipped, rationale_text = self._get_answer_span(answer, answer_type, paragraph_text)
                question_text = self._get_question_text(question_history, question)
                question_history = self._get_question_history(question_history, question, answer, answer_type, is_skipped, self.num_turn)
                
                if answer_type not in ["unknown", "yes", "no"] and not is_skipped and answer_text:
                    answer_span_text = self._process_found_answer(answer["input_text"], answer_text)
                else:
                    answer_span_text = ""
                    
                orig_answer_text = answer["input_text"]
                
                if answer_type in ["unknown", "yes", "no"]:
                    answer_span_text = answer_type
                
                if answer_type == "number":
                    answer_span_text = answer_subtype
                    
                if answer_type == "option":
                    answer_span_text = answer_subtype
                    
                if answer_span_text.lower() == orig_answer_text.lower():
                    label = 1
                else:
                    label = 0
                    
                example = eachExample(
                    qas_id=qas_id,
                    unique_id=unique_id,
                    question_text=question_text,
                    rationale_text=rationale_text,
                    answer_span_text=answer_span_text,
                    orig_answer_text=orig_answer_text,
                    label=label)

                each_examples.append(example)
                unique_id += 1
            input_example = InputExample(
                para_text=paragraph_text,
                examples=each_examples)
            input_examples.append(input_example)
                
        return input_examples

class XLNetTokenizer(object):
    """Default text tokenizer for XLNet"""
    def __init__(self,
                 sp_model_file,
                 lower_case=False):
        """Construct XLNet tokenizer"""
        self.sp_processor = sp.SentencePieceProcessor()
        self.sp_processor.Load(sp_model_file)
        self.lower_case = lower_case
    
    def tokenize(self,
                 text):
        """Tokenize text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        tokenized_pieces = prepro_utils.encode_pieces(self.sp_processor, processed_text, return_unicode=False)
        return tokenized_pieces
    
    def encode(self,
               text):
        """Encode text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        encoded_ids = prepro_utils.encode_ids(self.sp_processor, processed_text)
        return encoded_ids
    
    def token_to_id(self,
                    token):
        """Convert token to id for XLNet"""
        return self.sp_processor.PieceToId(token)
    
    def id_to_token(self,
                    id):
        """Convert id to token for XLNet"""
        return self.sp_processor.IdToPiece(id)
    
    def tokens_to_ids(self,
                      tokens):
        """Convert tokens to ids for XLNet"""
        return [self.sp_processor.PieceToId(token) for token in tokens]
    
    def ids_to_tokens(self,
                      ids):
        """Convert ids to tokens for XLNet"""
        return [self.sp_processor.IdToPiece(id) for id in ids]

class XLNetExampleProcessor(object):
    """Default example processor for XLNet"""
    def __init__(self,
                 doc_stride,
                 max_seq_length,
                 max_query_length,
                 tokenizer):
        """Construct XLNet example processor"""
        self.special_vocab_list = ["<unk>", "<s>", "</s>", "<cls>", "<sep>", "<pad>", "<mask>", "<eod>", "<eop>"]
        self.special_vocab_map = {}
        for (i, special_vocab) in enumerate(self.special_vocab_list):
            self.special_vocab_map[special_vocab] = i
        
        self.segment_vocab_list = ["<q>", "<c>", "<a>", "<cls>", "<sep>", "<pad>"]
        self.segment_vocab_map = {}
        for (i, segment_vocab) in enumerate(self.segment_vocab_list):
            self.segment_vocab_map[segment_vocab] = i
        
        self.doc_stride = doc_stride
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.tokenizer = tokenizer
        self.unique_id = 1000000000
    
    def _generate_match_mapping(self,
                                para_text,
                                tokenized_para_text,
                                N,
                                M,
                                max_N,
                                max_M):
        """Generate match mapping for raw and tokenized paragraph"""
        def _lcs_match(para_text,
                       tokenized_para_text,
                       N,
                       M,
                       max_N,
                       max_M,
                       max_dist):
            """longest common sub-sequence
            
            f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
            
            unlike standard LCS, this is specifically optimized for the setting
            because the mismatch between sentence pieces and original text will be small
            """
            f = np.zeros((max_N, max_M), dtype=np.float32)
            g = {}
            
            for i in range(N):
                for j in range(i - max_dist, i + max_dist):
                    if j >= M or j < 0:
                        continue
                    
                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]
                    
                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]
                    
                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    
                    raw_char = prepro_utils.preprocess_text(para_text[i], lower=self.tokenizer.lower_case, remove_space=False)
                    tokenized_char = tokenized_para_text[j]
                    if (raw_char == tokenized_char and f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1
            
            return f, g
        
        max_dist = abs(N - M) + 5
        for _ in range(2):
            lcs_matrix, match_mapping = _lcs_match(para_text, tokenized_para_text, N, M, max_N, max_M, max_dist)
            
            if lcs_matrix[N - 1, M - 1] > 0.8 * N:
                break
            
            max_dist *= 2
        
        mismatch = lcs_matrix[N - 1, M - 1] < 0.8 * N
        return match_mapping, mismatch
    
    def _convert_tokenized_index(self,
                                 index,
                                 pos,
                                 M=None,
                                 is_start=True):
        """Convert index for tokenized text"""
        if index[pos] is not None:
            return index[pos]
        
        N = len(index)
        rear = pos
        while rear < N - 1 and index[rear] is None:
            rear += 1
        
        front = pos
        while front > 0 and index[front] is None:
            front -= 1
        
        assert index[front] is not None or index[rear] is not None
        
        if index[front] is None:
            if index[rear] >= 1:
                if is_start:
                    return 0
                else:
                    return index[rear] - 1
            
            return index[rear]
        
        if index[rear] is None:
            if M is not None and index[front] < M - 1:
                if is_start:
                    return index[front] + 1
                else:
                    return M - 1
            
            return index[front]
        
        if is_start:
            if index[rear] > index[front] + 1:
                return index[front] + 1
            else:
                return index[rear]
        else:
            if index[rear] > index[front] + 1:
                return index[rear] - 1
            else:
                return index[front]
    
    def convert_coqa_example(self,
                             total_example,
                             logging=False):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        """
        para_text = total_example.para_text
        para_tokens = self.tokenizer.tokenize(para_text)
        
        char2token_index = []
        token2char_start_index = []
        token2char_end_index = []
        char_idx = 0
        for i, token in enumerate(para_tokens):
            char_len = len(token)
            char2token_index.extend([i] * char_len)
            token2char_start_index.append(char_idx)
            char_idx += char_len
            token2char_end_index.append(char_idx - 1)
        
        tokenized_para_text = ''.join(para_tokens).replace(prepro_utils.SPIECE_UNDERLINE, ' ')
        
        N, M = len(para_text), len(tokenized_para_text)
        max_N, max_M = 1024, 1024
        if N > max_N or M > max_M:
            max_N = max(N, max_N)
            max_M = max(M, max_M)
        
        match_mapping, mismatch = self._generate_match_mapping(para_text, tokenized_para_text, N, M, max_N, max_M)
        
        raw2tokenized_char_index = [None] * N
        tokenized2raw_char_index = [None] * M
        i, j = N-1, M-1
        while i >= 0 and j >= 0:
            if (i, j) not in match_mapping:
                break
            
            if match_mapping[(i, j)] == 2:
                raw2tokenized_char_index[i] = j
                tokenized2raw_char_index[j] = i
                i, j = i - 1, j - 1
            elif match_mapping[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1
        
        if all(v is None for v in raw2tokenized_char_index) or mismatch:
            tf.logging.warning("raw and tokenized paragraph mismatch detected for example: %s" % example.qas_id)
        
        token2char_raw_start_index = []
        token2char_raw_end_index = []
        for idx in range(len(para_tokens)):
            start_pos = token2char_start_index[idx]
            end_pos = token2char_end_index[idx]
            raw_start_pos = self._convert_tokenized_index(tokenized2raw_char_index, start_pos, N, is_start=True)
            raw_end_pos = self._convert_tokenized_index(tokenized2raw_char_index, end_pos, N, is_start=False)
            token2char_raw_start_index.append(raw_start_pos)
            token2char_raw_end_index.append(raw_end_pos)
        """
        
        feature_list = []
        passed_features = 0
        for example in total_example.examples:
            query_tokens = []
            qa_texts = example.question_text.split('<s>')
            for qa_text in qa_texts:
                qa_text = qa_text.strip()
                if not qa_text:
                    continue

                query_tokens.append('<s>')

                qa_items = qa_text.split('</s>')
                if len(qa_items) < 1:
                    continue

                q_text = qa_items[0].strip()
                q_tokens = self.tokenizer.tokenize(q_text)
                query_tokens.extend(q_tokens)

                if len(qa_items) < 2:
                    continue

                query_tokens.append('</s>')

                a_text = qa_items[1].strip()
                a_tokens = self.tokenizer.tokenize(a_text)
                query_tokens.extend(a_tokens)

            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[-self.max_query_length:]

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_para_length = self.max_seq_length - len(query_tokens) - 3

            input_tokens = []
            segment_ids = []
            input_tokens.append("<cls>")
            segment_ids.append(self.segment_vocab_map["<cls>"])

            # We put P before Q because during pretraining, B is always shorter than A
            for query_token in query_tokens:
                input_tokens.append(query_token)
                segment_ids.append(self.segment_vocab_map["<q>"])
            
            answer_span_text = example.answer_span_text
            """
            answer_start_char = para_text.find(answer_span_text)
            assert answer_start_char!=-1

            raw_start_char_pos = answer_start_char
            raw_end_char_pos = raw_start_char_pos + len(example.answer_span_text) - 1
            tokenized_start_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_start_char_pos, is_start=True)
            tokenized_end_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_end_char_pos, is_start=False)
            tokenized_start_token_pos = char2token_index[tokenized_start_char_pos]
            tokenized_end_token_pos = char2token_index[tokenized_end_char_pos]
            assert tokenized_start_token_pos <= tokenized_end_token_pos


            answer_start_token = tokenized_start_token_pos
            answer_end_token = tokenized_end_token_pos

            rest_stride = self.doc_stride - (answer_end_token-answer_start_token+1)
            fore_stride = int(rest_stride/2)
            rear_stride = int(rest_stride/2)
            context_tokens = []
            context_start_token = answer_start_token-fore_stride
            if context_start_token >= 0:
                context_tokens.extend(para_tokens[context_start_token:answer_start_token])
            else:
                context_tokens.extend(para_tokens[:answer_start_token])
            context_tokens.extend(para_tokens[answer_start_token:answer_end_token+1])
            context_end_token = answer_end_token+rear_stride
            if context_end_token < len(para_tokens):
                context_tokens.extend(para_tokens[answer_end_token:context_end_token])
            else:
                context_tokens.extend(para_tokens[answer_end_token:])

            for context_token in context_tokens:
                input_tokens.append(context_token)
                segment_ids.append(self.segment_vocab_map["<c>"])

    
            rationale_text = example.rationale_text
            rationale_tokens = self.tokenizer.tokenize(rationale_text)
            rationale_tokens = ["<s>"] + rationale_tokens + ["</s>"]
            for rationale_token in rationale_tokens:
                input_tokens.append(rationale_token)
                segment_ids.append(self.segment_vocab_map["<r>"])

            """
            input_tokens.append("<sep>")
            segment_ids.append(self.segment_vocab_map["<sep>"])

            answer_span_tokens = self.tokenizer.tokenize(answer_span_text)  
            for answer_token in answer_span_tokens:
                input_tokens.append(answer_token)
                segment_ids.append(self.segment_vocab_map["<a>"])

            input_tokens.append("<sep>")
            segment_ids.append(self.segment_vocab_map["<sep>"])

            input_ids = self.tokenizer.tokens_to_ids(input_tokens)

            # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
            input_mask = [0] * len(input_ids)

            if len(input_ids) > self.max_seq_length:
                continue

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(self.special_vocab_map["<pad>"])
                input_mask.append(1)
                segment_ids.append(self.segment_vocab_map["<pad>"])

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            label = example.label

            if logging:
                tf.logging.info("*** Example ***")
                tf.logging.info("qas_id: %s" % example.unique_id)
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                printable_input_tokens = [prepro_utils.printable_text(input_token) for input_token in input_tokens]
                tf.logging.info("input_tokens: %s" % input_tokens)
                tf.logging.info("label: %s" % str(label))

            feature = InputFeatures(
                qas_id=example.unique_id,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=label)

            feature_list.append(feature)
        
        return feature_list
    
    def convert_examples_to_features(self,
                                     examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        for (idx, example) in enumerate(examples):
            if idx % 1000 == 0:
                tf.logging.info("Converting example %d of %d" % (idx, len(examples)))

            feature_list = self.convert_coqa_example(example, logging=(idx < 20))
            features.extend(feature_list)
        
        tf.logging.info("Generate %d features from %d examples" % (len(features), len(examples)))
        
        return features
    
    def save_features_as_tfrecord(self,
                                  features,
                                  output_file):
        """Save a set of `InputFeature`s to a TFRecord file."""
        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        
        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        
        with tf.python_io.TFRecordWriter(output_file) as writer:
            for feature in features:
                features = collections.OrderedDict()
                features["qas_id"] = create_int_feature([feature.qas_id])
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_float_feature(feature.input_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["label"] = create_float_feature([feature.label])
                
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
    
    def save_features_as_pickle(self,
                                features,
                                output_file):
        """Save a set of `InputFeature`s to a Pickle file."""
        with open(output_file, 'wb') as file:
            pickle.dump(features, file)
    
    def load_features_from_pickle(self,
                                  input_file):
        """Load a set of `InputFeature`s from a Pickle file."""
        if not os.path.exists(input_file):
            raise FileNotFoundError("feature file not found: {0}".format(input_file))
        
        with open(input_file, 'rb') as file:
            features = pickle.load(file)
            return features

class XLNetInputBuilder(object):
    """Default input builder for XLNet"""
    @staticmethod
    def get_input_fn(input_file,
                     seq_length,
                     is_training,
                     drop_remainder,
                     shuffle_buffer=2048,
                     num_threads=16):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        name_to_features = {
            "qas_id": tf.FixedLenFeature([], tf.int64),
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64)
        }
        
        if is_training:
            name_to_features["label"] = tf.FixedLenFeature([], tf.float32)            
        
        def _decode_record(record,
                           name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)
            
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example
        
        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]
            
            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=shuffle_buffer, seed=np.random.randint(10000))
            
            d = d.apply(tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_threads,
                drop_remainder=drop_remainder))
            
            return d.prefetch(1024)
        
        return input_fn
    
    @staticmethod
    def get_serving_input_fn(seq_length):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        def serving_input_fn():
            with tf.variable_scope("serving"):
                features = {
                    'qas_id': tf.placeholder(tf.int32, [None], name='qas_id'),
                    'input_ids': tf.placeholder(tf.int32, [None, seq_length], name='input_ids'),
                    'input_mask': tf.placeholder(tf.float32, [None, seq_length], name='input_mask'),
                    'segment_ids': tf.placeholder(tf.int32, [None, seq_length], name='segment_ids')
                }
                
                return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()
        
        return serving_input_fn

class XLNetModelBuilder(object):
    """Default model builder for XLNet"""
    def __init__(self,
                 model_config,
                 use_tpu=False):
        """Construct XLNet model builder"""
        self.model_config = model_config
        self.use_tpu = use_tpu
      
    def _create_model(self,
                      is_training,
                      input_ids,
                      input_mask,
                      segment_ids,
                      label=None):
        """Creates XLNet-CoQA model"""

        model = xlnet.XLNetModel(
            xlnet_config=self.model_config,
            run_config=xlnet.create_run_config(is_training, True, FLAGS),
            input_ids=tf.transpose(input_ids, perm=[1,0]),                                                               # [b,l] --> [l,b]
            input_mask=tf.transpose(input_mask, perm=[1,0]),                                                             # [b,l] --> [l,b]
            seg_ids=tf.transpose(segment_ids, perm=[1,0]))                                                               # [b,l] --> [l,b]
        
        initializer = model.get_initializer()
        seq_len = tf.shape(input_ids)[-1]
        output_result = tf.transpose(model.get_sequence_output(), perm=[1,0,2])                                      # [l,b,h] --> [b,l,h]
        cls_result = output_result[:,0,:]

        with tf.variable_scope("nlu", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
                result = tf.layers.dense(cls_result, units=self.model_config.d_model, activation=tf.tanh,
                                         use_bias=True, kernel_initializer=initializer, bias_initializer=tf.zeros_initializer,
                                         kernel_regularizer=None, bias_regularizer=None, trainable=True, name="classification1")
                result = tf.contrib.layers.layer_norm(result, center=True, scale=True, activation_fn=None,
                        reuse=None, begin_norm_axis=-1, begin_params_axis=-1, trainable=True, scope="norm")
                result = tf.layers.dense(result, units=1, activation=None,
                                         use_bias=True, kernel_initializer=initializer, bias_initializer=tf.zeros_initializer,
                                         kernel_regularizer=None, bias_regularizer=None, trainable=True, name="classification2")
                
                result = tf.squeeze(result)
                prob = tf.sigmoid(result)
               
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
                loss = tf.constant(0.0, dtype=tf.float32)
                if is_training:
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=result)
                    loss = tf.reduce_mean(loss)
   
        return loss, prob
    
    def get_model_fn(self):
        """Returns `model_fn` closure for TPUEstimator."""
        def model_fn(features,
                     labels,
                     mode,
                     params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            
            qas_id = features["qas_id"]
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            
            if is_training:
                label = features["label"]
                    
            else:
                label = None
            
            loss, prob = self._create_model(is_training, input_ids, input_mask, segment_ids, label)
                
            scaffold_fn = model_utils.init_from_checkpoint(FLAGS)
            
            output_spec = None
            if is_training:
                train_op, _, _ = model_utils.get_train_op(FLAGS, loss)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={
                        "qas_id": qas_id,
                        "prob": prob},
                    scaffold_fn=scaffold_fn)
            
            return output_spec
        
        return model_fn

class XLNetPredictProcessor(object):
    """Default predict processor for XLNet"""
    def __init__(self,
                 output_dir,
                 tokenizer,
                 predict_tag=None):
        """Construct XLNet predict processor"""
        self.tokenizer = tokenizer
        
        predict_tag = predict_tag if predict_tag else str(time.time())
        self.output_summary = os.path.join(output_dir, "predict.{0}.json".format(predict_tag))
    
    def _write_to_json(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:  
            json.dump(data_list, file, indent=4)
    
    def _write_to_text(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:
            for data in data_list:
                file.write("{0}\n".format(data))
    
    def process(self,
                total_examples,
                results):
        predict_summary_list = []
        
        qas_id_to_results = {}
        for result in results:
            qas_id_to_results[result.qas_id] = result
            
        y_true = []
        y_pred = []
            
        num_example = 0
        for example_list in total_examples:
            num_example+=len(example_list.examples)
            for (example_idx, example) in enumerate(example_list.examples):
                if example_idx % 1000 == 0:
                    tf.logging.info('Updating {0} example with predict'.format(example_idx))

                if example.unique_id not in qas_id_to_results:
                    tf.logging.warning('No result found for example: {0}'.format(example.unique_id))
                    continue

                qas_id = example.unique_id
                result = qas_id_to_results[qas_id]

                qas_id = example.qas_id
                query = example.question_text
                rationale = example.rationale_text
                answer = example.answer_span_text
                gold_answer = example.orig_answer_text
                probability = float(result.prob)
                prediction = 1 if probability >= 0.5 else 0
                label = example.label

                y_true.append(label)
                y_pred.append(prediction)

                predict_summary_list.append({
                    "qas_id": example.qas_id,
                    "question_text": query,
                    "rationale": rationale,
                    "extracted_answer": answer,
                    "original_answer": gold_answer,
                    "predicted_label": prediction,
                    "label": label,
                    "probability": probability
                })
        
        self._write_to_json(predict_summary_list, self.output_summary)
        
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    np.random.seed(FLAGS.random_seed)
    
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    
    task_name = FLAGS.task_name.lower()
    data_pipeline = CoqaPipeline(
        data_dir=FLAGS.data_dir,
        task_name=task_name,
        num_turn=FLAGS.num_turn)
    
    model_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    
    model_builder = XLNetModelBuilder(
        model_config=model_config,
        use_tpu=FLAGS.use_tpu)
    
    model_fn = model_builder.get_model_fn()
    
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    tpu_config = model_utils.configure_tpu(FLAGS)
    
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=tpu_config,
        export_to_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    
    tokenizer = XLNetTokenizer(
        sp_model_file=FLAGS.spiece_model_file,
        lower_case=FLAGS.lower_case)
    
    example_processor = XLNetExampleProcessor(
        doc_stride=FLAGS.doc_stride,
        max_seq_length=FLAGS.max_seq_length,
        max_query_length=FLAGS.max_query_length,
        tokenizer=tokenizer)
    
    if FLAGS.do_train:
        train_examples = data_pipeline.get_train_examples()
        
        tf.logging.info("***** Run training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", FLAGS.train_steps)
        
        train_record_file = os.path.join(FLAGS.output_dir, "train-{0}.tfrecord".format(task_name))
        if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
            train_features = example_processor.convert_examples_to_features(train_examples)
            np.random.shuffle(train_features)
            example_processor.save_features_as_tfrecord(train_features, train_record_file)
        
        train_input_fn = XLNetInputBuilder.get_input_fn(train_record_file, FLAGS.max_seq_length, True, True, FLAGS.shuffle_buffer)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_predict:
        predict_examples = data_pipeline.get_dev_examples()
        
        tf.logging.info("***** Run prediction *****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        
        predict_record_file = os.path.join(FLAGS.output_dir, "dev-{0}.tfrecord".format(task_name))
        predict_pickle_file = os.path.join(FLAGS.output_dir, "dev-{0}.pkl".format(task_name))
        if not os.path.exists(predict_record_file) or not os.path.exists(predict_pickle_file) or FLAGS.overwrite_data:
            predict_features = example_processor.convert_examples_to_features(predict_examples)
            example_processor.save_features_as_tfrecord(predict_features, predict_record_file)
            example_processor.save_features_as_pickle(predict_features, predict_pickle_file)
        else:
            predict_features = example_processor.load_features_from_pickle(predict_pickle_file)

        predict_input_fn = XLNetInputBuilder.get_input_fn(predict_record_file, FLAGS.max_seq_length, False, False)
        results = estimator.predict(input_fn=predict_input_fn)
        
        predict_results = [OutputResult(
            qas_id=result["qas_id"],
            prob=result["prob"]
        ) for result in results]
        
        predict_processor = XLNetPredictProcessor(
            FLAGS.output_dir,
            tokenizer=tokenizer,
            predict_tag=FLAGS.predict_tag)
        
        predict_processor.process(predict_examples, predict_results)
        
    if FLAGS.do_export:
        tf.logging.info("***** Running exporting *****")
        if not os.path.exists(FLAGS.export_dir):
            os.mkdir(FLAGS.export_dir)
        
        serving_input_fn = XLNetInputBuilder.get_serving_input_fn(FLAGS.max_seq_length)
        estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn, as_text=False)

if __name__ == "__main__":
    flags.mark_flag_as_required("spiece_model_file")
    flags.mark_flag_as_required("model_config_path")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("export_dir")
    tf.app.run()
