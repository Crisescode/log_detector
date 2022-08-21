# coding: utf-8

import os
import re
import math
import random
import hashlib
import operator
import pandas as pd
from typing import List
from datetime import datetime


class Params:
    def __init__(self, path: str, rex: List, save_path: str, group_num: int, log_format: str):
        self.path = path
        self.rex = rex
        self.savePath = save_path
        self.groupNum = group_num
        self.log_format = log_format


class DataCleaner:
    def __init__(self, input_path: str, output_path: str, group_num: int, log_format: str, rex: List = [], seed=0):
        self.params = Params(
            path=input_path,
            rex=rex,
            save_path=output_path,
            group_num=group_num,
            log_format=log_format
        )
        self.wordLL = []
        self.logline_num = []
        self.termpairLLT = []
        self.logNumPerGroup = []
        self.termPairLogNumLD = []
        self.logIndexPerGroup = []
        self.groupIndex = dict()
        self.seed = seed

    def generate_log_format_rex(self, log_format):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', log_format)

        print("splitters", splitters)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def log_to_dataframe(self, log_file: str, regex, headers: List[str]):
        log_message = []
        line_count = 0
        with open(log_file, "r") as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_message.append(message)
                    line_count += 1
                except Exception as e:
                    print(e)
                    pass

        log_df = pd.DataFrame(log_message, columns=headers)
        log_df.insert(0, "LineId", None)
        log_df['LineId'] = [i + 1 for i in range(line_count)]
        return log_df

    def load_log(self):
        print("Loading logs ...")
        headers, rex = self.generate_log_format_rex(self.params.log_format)

        self.df_log = self.log_to_dataframe(
            log_file=os.path.join(self.params.path, self.log_name),
            regex=rex,
            headers=headers
        )

        for idx, line in self.df_log.iterrows():
            line = line['Content']
            if self.params.rex:
                for currentRex in self.params.rex:
                    line = re.sub(currentRex, '', line)

            wordSeq = line.strip().split()
            self.wordLL.append(tuple(wordSeq))

    def generate_term_pair(self):
        print("Generating term pair ...")
        i = 0
        for wordL in self.wordLL:
            wordLT = []
            for j in range(len(wordL)):
                for k in range(j + 1, len(wordL), 1):
                    if wordL[j] != "[$]" and wordL[k] != "[$]":
                        term_pair = (wordL[j], wordL[k])
                        wordLT.append(term_pair)
            self.termpairLLT.append(wordLT)
            i += 1

        for i in range(self.params.groupNum):
            newDict = dict()
            self.termPairLogNumLD.append(newDict)
            # initialize the item value to zero
            self.logNumPerGroup.append(0)

        self.loglineNum = len(self.wordLL)
        random.seed(self.seed)
        for i in range(self.loglineNum):
            ran = random.randint(0, self.params.groupNum - 1)
            self.groupIndex[i] = ran
            self.logNumPerGroup[ran] += 1

        i = 0
        for termpairLT in self.termpairLLT:
            j = 0
            for key in termpairLT:
                currGroupIndex = self.groupIndex[i]
                if key not in self.termPairLogNumLD[currGroupIndex]:
                    self.termPairLogNumLD[currGroupIndex][key] = 1
                else:
                    self.termPairLogNumLD[currGroupIndex][key] += 1
                j += 1
            i += 1

    def LogMessParti(self):
        print('Log message partitioning...')
        changed = True
        while changed:
            changed = False
            i = 0
            for termpairLT in self.termpairLLT:
                curGroup = self.groupIndex[i]
                alterGroup = format_func(curGroup, self.termPairLogNumLD, self.logNumPerGroup, i, termpairLT,
                                         self.params.groupNum)
                if curGroup != alterGroup:
                    changed = True
                    self.groupIndex[i] = alterGroup
                    # update the dictionary of each group
                    for key in termpairLT:
                        # minus 1 from the current group count on this key
                        self.termPairLogNumLD[curGroup][key] -= 1
                        if self.termPairLogNumLD[curGroup][key] == 0:
                            del self.termPairLogNumLD[curGroup][key]
                        # add 1 to the alter group
                        if key not in self.termPairLogNumLD[alterGroup]:
                            self.termPairLogNumLD[alterGroup][key] = 1
                        else:
                            self.termPairLogNumLD[alterGroup][key] += 1
                    self.logNumPerGroup[curGroup] -= 1
                    self.logNumPerGroup[alterGroup] += 1
                i += 1

    def signatConstr(self):
        print('Log message signature construction...')
        if not os.path.exists(self.params.savePath):
            os.makedirs(self.params.savePath)

        wordFreqPerGroup = []
        candidateTerm = []
        candidateSeq = []
        self.signature = []

        for t in range(self.params.groupNum):
            dic = dict()
            newlogIndex = []
            newCandidate = dict()
            wordFreqPerGroup.append(dic)
            self.logIndexPerGroup.append(newlogIndex)
            candidateSeq.append(newCandidate)

        lineNo = 0
        for wordL in self.wordLL:
            groupIndex = self.groupIndex[lineNo]
            self.logIndexPerGroup[groupIndex].append(lineNo)
            for key in wordL:
                if key not in wordFreqPerGroup[groupIndex]:
                    wordFreqPerGroup[groupIndex][key] = 1
                else:
                    wordFreqPerGroup[groupIndex][key] += 1
            lineNo += 1

        for i in range(self.params.groupNum):
            halfLogNum = math.ceil(self.logNumPerGroup[i] / 2.0)
            dic = dict((k, v) for k, v in wordFreqPerGroup[i].items() if v >= halfLogNum)
            candidateTerm.append(dic)

        lineNo = 0
        for wordL in self.wordLL:
            curGroup = self.groupIndex[lineNo]
            newCandiSeq = []

            for key in wordL:
                if key in candidateTerm[curGroup]:
                    newCandiSeq.append(key)

            keySeq = tuple(newCandiSeq)
            if keySeq not in candidateSeq[curGroup]:
                candidateSeq[curGroup][keySeq] = 1
            else:
                candidateSeq[curGroup][keySeq] += 1
            lineNo += 1

        for i in range(self.params.groupNum):
            sig = max(candidateSeq[i].items(), key=operator.itemgetter(1))[0]
            self.signature.append(sig)

    def writeResultToFile(self):
        idx_eventID = {}
        for idx, item in enumerate(self.signature):
            eventStr = ' '.join(item)
            idx_eventID[idx] = hashlib.md5(eventStr.encode('utf-8')).hexdigest()[0:8]

        EventId = []
        EventTemplate = []
        LineId_groupId = []
        for idx, item in enumerate(self.logIndexPerGroup):
            for LineId in item:
                LineId_groupId.append([LineId, idx])
        LineId_groupId.sort(key=lambda x:x[0])
        for item in LineId_groupId:
            GroupID = item[1]
            EventId.append(idx_eventID[GroupID])
            EventTemplate.append(' '.join(self.signature[GroupID]))

        self.df_log['EventId'] = EventId
        self.df_log['EventTemplate'] = EventTemplate
        self.df_log.to_csv(os.path.join(self.params.savePath, self.log_name + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)

        df_event.to_csv(os.path.join(
            self.params.savePath, self.log_name + '_templates.csv'),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"]
        )

    def analyze(self, log_name: str):
        print("Parsing file: " + os.path.join(self.params.path, log_name))
        self.log_name = log_name
        start_time = datetime.now()
        self.load_log()
        self.generate_term_pair()
        self.LogMessParti()
        self.signatConstr()
        self.writeResultToFile()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))


def format_func(curGroupIndex, termPairLogNumLD, logNumPerGroup, lineNum, termpairLT, k):
    maxDeltaD = 0
    maxJ = curGroupIndex
    for i in range(k):
        returnedDeltaD = get_deltaD(logNumPerGroup, termPairLogNumLD, curGroupIndex, i, lineNum, termpairLT)
        if returnedDeltaD > maxDeltaD:
            maxDeltaD = returnedDeltaD
            maxJ = i
    return maxJ


def get_deltaD(logNumPerGroup, termPairLogNumLD, groupI, groupJ, lineNum, termpairLT):
    deltaD = 0
    Ci = logNumPerGroup[groupI]
    Cj = logNumPerGroup[groupJ]
    for r in termpairLT:
        if r in termPairLogNumLD[groupJ]:
            deltaD += (pow(((termPairLogNumLD[groupJ][r] + 1) / (Cj + 1.0)), 2)
                       - pow((termPairLogNumLD[groupI][r] / (Ci + 0.0)), 2))
        else:
            deltaD += (pow((1 / (Cj + 1.0)), 2) - pow((termPairLogNumLD[groupI][r] / (Ci + 0.0)), 2))
    deltaD = deltaD * 3
    return deltaD


if __name__ == "__main__":
    input_dir = "./data"
    output_dir = "./data/clean_result"
    data_clean_settings = {
        'HDFS': {
            'log_file': 'data_2k.log',
            'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
            'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
            'group_num': 15
        }
    }

    for dataset, setting in data_clean_settings.items():
        print("\n=== Evaluation on %s ===" % dataset)
        in_dir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
        log_file = os.path.basename(setting["log_file"])

        parser = DataCleaner(
            log_format=setting["log_format"],
            input_path=input_dir,
            output_path=output_dir,
            rex=setting["regex"],
            group_num=setting["group_num"]
        )
        parser.analyze(log_file)
