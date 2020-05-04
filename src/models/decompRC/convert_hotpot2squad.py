import os
import json
import argparse

import numpy as np
from tqdm import tqdm
from prepro_util import find_span_from_text

title_s = "<title>"
title_e = "</title>"

BASE_DIR = "/scratch/tabhishek/decomprc"

def save(data, dir_name, data_type):
    if not os.path.isdir(os.path.join(BASE_DIR, 'data', dir_name)):
        os.makedirs(os.path.join(BASE_DIR, 'data', dir_name))

    file_path = os.path.join(BASE_DIR, 'data', dir_name, '{}.json'.format(data_type))
    with open(file_path, 'w') as f:
        print ("Saving {}".format(file_path))
        json.dump({'data': data}, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='hotpotqa')
    parser.add_argument('--task', type=str, default="hotpot-all")
    args = parser.parse_args()

    if args.task == 'hotpot-all':
        training_data = load_hotpot(args, 'train')
        save(training_data, 'hotpot-all', 'train')
        dev_data = load_hotpot(args,  'dev_distractor')
        save(dev_data, 'hotpot-all', 'dev')
    else:
        raise NotImplementedError()

def load_hotpot(args, data_type, only_bridge=False, only_comparison=False,
                only_sf=False, only_gold=False):
    with open(os.path.join(args.data_dir, "hotpot_{}_v1.json".format(data_type)), 'r') as f:
        data = json.load(f)

    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    data_list = []
    n_paras = []
    n_gold_paras = []
    n_paras_with_answer = []
    n_sents = []
    n_answers = []
    no_answer = 0
    acc_list = {'overall': [], 'comparison':[], 'bridge':[]}

    for article_id, article in tqdm(enumerate(data)): 
        if only_bridge and article['type'] != 'bridge':
            continue
        if only_comparison and article['type'] != 'comparison':
            continue
        paragraphs = article['context']
        sfs = [(_process_sent(t), s) for t, s in article['supporting_facts']]
        question = article['question']
        answer = article['answer'].strip()

        para_with_sf = set()
        contexts_list, answers_list = [], []
        for para_idx, para in enumerate(paragraphs):
            #10 or less articles
            title = _process_sent(para[0])
            content = para[1]
            answers = []
            contexts = ["{} {} {}".format(title_s, title.lower().strip(), title_e)]
            offset = len(contexts[0]) + 1

            if only_gold and title not in [t for t, _ in sfs]:
                continue
            #for each article, associated paragraphs are been scaned
            for sent_idx, sent in enumerate(content):
                is_sf = (title, sent_idx) in sfs
                if only_sf and not is_sf:
                    continue
                contexts.append(sent.lower().strip())
                if is_sf:
                    para_with_sf.add(para_idx)
                    if answer in ['yes', 'no']:
                        answers.append({'text': answer, 'answer_start': -1})
                    elif answer.lower() in contexts[-1]:
                        assert contexts[-1] == sent.lower().strip()
                        curr_answers = find_span_from_text(contexts[-1], contexts[-1].split(' '), answer.lower())
                        for i, curr_answer in enumerate(curr_answers):
                            curr_answers[i]['answer_start'] += offset
                        answers += curr_answers
                offset += len(contexts[-1]) + 1

            if len(contexts)>1:
                n_sents.append(len(contexts))
                context = " ".join(contexts)
                contexts_list.append(context)
                answers_list.append(answers)

        assert len(para_with_sf)>1
        assert len(contexts_list)>1

        if only_sf:
            merged_context = ""
            merged_answers = []
            offset = 0
            for (context, answers) in zip(contexts_list, answers_list):
                for i, a in enumerate(answers):
                    answers[i]['answer_start'] += len(merged_context)
                merged_context += context + " "
                merged_answers += answers
            contexts_list, answers_list = [merged_context], [merged_answers]

        assert len(contexts_list)==len(answers_list)
        n_paras.append(len(contexts_list))
        n_gold_paras.append(len(para_with_sf))
        n_paras_with_answer.append(len([a for a in answers_list if len(a)>0]))

        for (context, answers) in zip(contexts_list, answers_list):
            for a in answers:
                if a['text'] not in ['yes', 'no']:
                    assert a['text'] == context[a['answer_start']:a['answer_start']+len(a['text'])]

        n_answers.append(sum([len(answers) for answers in answers_list]))
        if n_answers[-1] == 0:
            no_answer += 1

        paragraph = {
                'context': contexts_list,
                'qas': [{
                    'final_answers': [answer],
                    'question': question,
                    'answers': answers_list,
                    'id': article['_id'],
                    'type': article['type']
                }]
            }
        data_list.append({'title': '', 'paragraphs': [paragraph]})

    print ("We have {}/{} number ({} with no answer) of HOTPOT examples!".format(len(data_list), len(data), no_answer))
    print ("On average, # paras = %.2f (%.2f gold and %.2f with answer ) / # sentences = %.2f / # answers = %.2f" % \
           (np.mean(n_paras), np.mean(n_gold_paras), np.mean(n_paras_with_answer), np.mean(n_sents), np.mean(n_answers)))

    return data_list


if __name__ == '__main__':
    main()
