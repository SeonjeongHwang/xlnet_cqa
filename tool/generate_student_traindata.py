import argparse
import json
import os
import collections

import tensorflow as tf
import numpy as np

def add_arguments(parser):
    parser.add_argument("--teacher_logits_dir", help="path to teacher's logits tfrecord files", required=True, type=str)
    parser.add_argument("--teacher_seeds", help="the teacher logits file list to knowledge distillation", required=True,
                        default="100", type=str)
    parser.add_argument("--student_train_file", help="path to student's train tfrecord file", required=True, type=str)
    parser.add_argument("--tag", help="tag for grouping several teachers", required=True, default="xxxxxx", type=str)
    parser.add_argument("--max_seq_length", help="Max sequence length", default=512, type=int)
    
def get_teacher_labels(teacher_files, seq_len):
    """
    output_logits = {
        "unique_id": tf.FixedLenFeature([], tf.int64),
        "unk_logits": tf.FixedLenFeature([seq_len], tf.float64),
        "yes_logits": tf.FixedLenFeature([seq_len], tf.float64),
        "no_logits": tf.FixedLenFeature([seq_len], tf.float64),
        "num_logits": tf.FixedLenFeature([seq_len], tf.float64),
        "opt_logits": tf.FixedLenFeature([seq_len], tf.float64),
        "start_logits": tf.FixedLenFeature([seq_len], tf.float64),
        "end_logits": tf.FixedLenFeature([seq_len], tf.float64)
    }
    """
    
    total_features = dict()
    for file in teacher_files:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        flag = True
        for record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(record)
            feature = dict(example.features.feature)
            
            if flag:
                print(feature)
                flag = False
            
            unique_id = int(np.array(feature["unique_id"].int64_list.value[0]))
            unk_logits = np.array(feature["unk_logits"].float_list.value[:])
            yes_logits = np.array(feature["yes_logits"].float_list.value[:])
            no_logits = np.array(feature["no_logits"].float_list.value[:])
            num_logits = np.array(feature["num_logits"].float_list.value[:])
            opt_logits = np.array(feature["opt_logits"].float_list.value[:])
            start_logits = np.array(feature["start_logits"].float_list.value[:])
            end_logits = np.array(feature["end_logits"].float_list.value[:])
            
            if unique_id in total_features:
                total_features[unique_id]["unk_logits"] += unk_logits
                total_features[unique_id]["yes_logits"] += yes_logits
                total_features[unique_id]["no_logits"] += no_logits
                total_features[unique_id]["num_logits"] += num_logits
                total_features[unique_id]["opt_logits"] += opt_logits
                total_features[unique_id]["start_logits"] += start_logits
                total_features[unique_id]["end_logits"] += end_logits
            else:
                logits = {
                    "unk_logits": unk_logits,
                    "yes_logits": yes_logits,
                    "no_logits": no_logits,
                    "num_logits": num_logits,
                    "opt_logits": opt_logits,
                    "start_logits": start_logits,
                    "end_logits": end_logits
                }
                total_features[unique_id] = logits
    
    total_teacher_num = len(teacher_seeds)
    for unique_id in total_features.keys():
        total_features[unique_id]["unk_logits"] /= total_teacher_num
        total_features[unique_id]["yes_logits"] /= total_teacher_num
        total_features[unique_id]["no_logits"] /= total_teacher_num
        total_features[unique_id]["num_logits"] /= total_teacher_num
        total_features[unique_id]["opt_logits"] /= total_teacher_num
        total_features[unique_id]["start_logits"] /= total_teacher_num
        total_features[unique_id]["end_logits"] /= total_teacher_num
        
    print("teacher's features: {0}".format(len(total_features)))
        
    return total_features
    
def load_student_features(student_file_dir):
    student_features = dict()
    
    record_iterator = tf.python_io.tf_record_iterator(path=student_file_dir)
    flag = True
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = dict(example.features.feature)
        
        if flag:
            print(feature)
            flag = False
        
        unique_id = feature["unique_id"].int64_list.value[0]
        input_ids = feature["input_ids"].int64_list.value[:]
        input_mask = feature["input_mask"].float_list.value[:]
        p_mask = feature["p_mask"].float_list.value[:]
        segment_ids = feature["segment_ids"].int64_list.value[:]
        rationale = feature["rationale"].int64_list.value[:]
        cls_index = feature["cls_index"].int64_list.value[0]
        start_position = feature["start_position"].int64_list.value[0]
        end_position = feature["end_position"].int64_list.value[0]
        is_unk = feature["is_unk"].float_list.value[0]
        is_yes = feature["is_yes"].float_list.value[0]
        is_no = feature["is_no"].float_list.value[0]
        number = feature["number"].float_list.value[0]
        option = feature["option"].float_list.value[0]
        
        student_features[unique_id] = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "p_mask": p_mask,
            "segment_ids": segment_ids,
            "rationale": rationale,
            "cls_index": cls_index,
            "start_position": start_position,
            "end_position": end_position,
            "is_unk": is_unk,
            "is_yes": is_yes,
            "is_no": is_no,
            "number": number,
            "option": option
        }
        
    print("student's features: {0}".format(len(student_features)))
    return student_features

def combine_and_write_tfrecord(logits, features, output_file):
    unique_ids = logits.keys()
    assert len(unique_ids)==len(features.keys())
    
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    def create_float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for unique_id in unique_ids:
            student_features = collections.OrderedDict()
            student_features["unique_id"] = create_int_feature(np.array([unique_id]))
            student_features["input_ids"] = create_int_feature(features[unique_id]["input_ids"])
            student_features["input_mask"] = create_float_feature(features[unique_id]["input_mask"])
            student_features["p_mask"] = create_float_feature(features[unique_id]["p_mask"])
            student_features["segment_ids"] = create_int_feature(features[unique_id]["segment_ids"])
            student_features["rationale"] = create_int_feature(features[unique_id]["rationale"])
            student_features["cls_index"] = create_int_feature([features[unique_id]["cls_index"]])

            student_features["start_position"] = create_int_feature([features[unique_id]["start_position"]])
            student_features["end_position"] = create_int_feature([features[unique_id]["end_position"]])
            student_features["is_unk"] = create_float_feature([features[unique_id]["is_unk"]])
            student_features["is_yes"] = create_float_feature([features[unique_id]["is_yes"]])
            student_features["is_no"] = create_float_feature([features[unique_id]["is_no"]])
            student_features["number"] = create_float_feature([features[unique_id]["number"]])
            student_features["option"] = create_float_feature([features[unique_id]["option"]])
            
            student_features["unk_logits"] = create_float_feature(logits[unique_id]["unk_logits"].tolist())
            student_features["yes_logits"] = create_float_feature(logits[unique_id]["yes_logits"].tolist())
            student_features["no_logits"] = create_float_feature(logits[unique_id]["no_logits"].tolist())
            student_features["num_logits"] = create_float_feature(logits[unique_id]["num_logits"].tolist())
            student_features["opt_logits"] = create_float_feature(logits[unique_id]["opt_logits"].tolist())
            student_features["start_logits"] = create_float_feature(logits[unique_id]["start_logits"].tolist())
            student_features["end_logits"] = create_float_feature(logits[unique_id]["end_logits"].tolist())
    
            tf_example = tf.train.Example(features=tf.train.Features(feature=student_features))
            writer.write(tf_example.SerializeToString())
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    
    teacher_seeds = args.teacher_seeds.strip().split(",")
    teacher_files = []
    for seed in teacher_seeds:
        f_name = os.path.join(args.teacher_logits_dir, "teacher_logits.{0}_{1}.tfrecord".format(args.tag, seed))
        assert os.path.exists(f_name), "{0} does not exist".format(f_name)
        teacher_files.append(f_name)
        
    ### teacher file들 읽어서 logits들 평균내기
    teacher_labels = get_teacher_labels(teacher_files, args.max_seq_length)
    
    ### student file 읽기
    student_features = load_student_features(args.student_train_file)
    
    ### logits과 student unique_id 기준으로 합치고 저장
    output_file = os.path.join(args.teacher_logits_dir, "student_{0}.train-coqa.tfrecord".format(args.tag))
    combine_and_write_tfrecord(teacher_labels, student_features, output_file)
