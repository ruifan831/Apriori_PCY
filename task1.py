from pyspark import SparkContext
import json
import itertools
import time
import sys
from collections import defaultdict

def singleFrequents(baskets,t):
    singles=defaultdict(int)
    for i in baskets:
        for j in i:
            singles[j] +=1
    result=get_fre_set(singles,t)
    return result


def getFrequentsOfSpecifiedSize(baskets,pre_fre,t,size):
    temp=defaultdict(int)
    candidate=combination(pre_fre,size)
    for i in candidate:
        key=tuple(sorted(i))
        for j in baskets:
            if set(i).issubset(j):
                temp[key] +=1
    result=get_fre_set(temp,t)
    return result
                
def hash_to_buckets(baskets,t):
    bit_map=defaultdict(int)
    for i in baskets:
        temp = itertools.combinations(sorted(i),2)
        for j in temp:
            key = hash_func(j)
            bit_map[key] +=1
    for k,v in bit_map.items():
        if v >=t:
            bit_map[k]=1
        else:
            bit_map[k]=0
    return bit_map

def hash_func(candidate):
    return hash(candidate)%10000

def combination(sets,size):
    if size <= 2: 
        return list(itertools.combinations(sets, size))
    else:
        return list(itertools.combinations(set(a for b in sets for a in b),size))
    
def count(baskets,candidates):
    temp=defaultdict(int)
    for i in candidates:
        for j in baskets:
            if set(i).issubset(j):
                temp[i] +=1
    return temp

def get_fre_set(setCount,t):
    result=[]
    for k,v in setCount.items():
        if v >= t:
            result.append(k)
    return sorted(result)
            
    
def apriori(iterator):
    baskets = []
    for i in iterator:
        baskets.append(i[1])
    proportion=len(baskets)/total
    temp_threshold=proportion*threshold
    frequents=[]
    fre_1_sets=singleFrequents(baskets,temp_threshold)
    frequents.extend(fre_1_sets)
    current_size = 2
    fre_2_candidates = combination(fre_1_sets,current_size)
    bitmap = hash_to_buckets(baskets,temp_threshold)
    temp_fre_2_sets=[]
    for i in fre_2_candidates:
        key = hash_func(i)
        if bitmap[key] == 1:
            temp_fre_2_sets.append(i)
    fre_2_dict = count(baskets,temp_fre_2_sets)
    fre_2_sets = get_fre_set(fre_2_dict,temp_threshold)
    frequents.extend(fre_2_sets)
    current_size += 1
    current_frequents=fre_2_sets
    while current_frequents:
        frequents_before=current_frequents
#         buckets=hash_to_buckets(baskets,temp_threshold,current_size)
        current_frequents= getFrequentsOfSpecifiedSize(baskets,frequents_before,temp_threshold,current_size)
        frequents.extend(current_frequents)
        current_size+=1
    return frequents

def main(caseNum,support,input_file,output_file):
    try:
        start=time.time()
        sc = SparkContext(appName="task1")
        lines = sc.textFile(input_file)
        global total,threshold
        threshold = support
        header = lines.first() 
        data = lines.filter(lambda x:x != header)
        if int(caseNum) == 1:
#             x = data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1])).groupByKey().map(lambda x:(x[0],set(x[1]))).filter(lambda x:len(x[1])>=70).cache()
            x = data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1])).groupByKey().map(lambda x:(x[0],set(x[1]))).cache()

        else:
            x = data.map(lambda x: x.split(",")).map(lambda x: (x[1],x[0])).groupByKey().cache()

        total = x.count()
        candidates=x.mapPartitions(apriori).distinct().map(lambda x: tuple([x]) if type(x) != tuple else x).sortBy(lambda x: (len(x),x)).collect()

        tempdata=x.map(lambda x: list(x[1])).collect()
        with open(output_file, "w") as f:
            result=[]
            f.write("Candidates:\n")
            for i in candidates:
                count=0
                for j in tempdata:
                    if set(i).issubset(j):
                        count+=1
                if count>=threshold:
                    result.append(i)
            f.write(",".join(map(lambda x: str(x).replace(",",""),filter(lambda x: len(x)==1,candidates))))
            prev_size=1
            for i in filter(lambda x:len(x)!=1,candidates):
                if len(i)>prev_size:
                    f.write("\n\n"+str(i))
                    prev_size=len(i)
                else:
                    f.write(","+str(i))
                    prev_size=len(i)
            f.write("\n\nFrequent Itemsets:\n")
            f.write(",".join(map(lambda x: str(x).replace(",",""),filter(lambda x: len(x)==1,result))))
            prev_size=1
            for i in filter(lambda x:len(x)!=1,result):
                if len(i)>prev_size:
                    f.write("\n\n"+str(i))
                    prev_size=len(i)
                else:
                    f.write(","+str(i))
                    prev_size=len(i)

        print(f"Duration: {time.time()-start}")
        sc.stop()
    except Exception as e:
        print(e)
        sc.stop()

if __name__ =="__main__":
    case = sys.argv[1]
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    main(case,support,input_file,output_file)