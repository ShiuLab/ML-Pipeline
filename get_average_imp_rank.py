### get mean importance rank among many imp files
import os,sys,numpy
 
start_dir= sys.argv[1]

def byScore(pair):
    return pair[1]

def get_dicts(inp, name, D):
    D2={}
    for line in inp:
        L= line.strip().split('\t')
        #print(L)
        cre= L[0]
        score= float(L[1])
        D2[cre]=score
    items= list(D2.items())
    items.sort()
    items.sort(key=byScore, reverse=True)
    n= len(items)
    #sorted_by_value = sorted(D2, key=lambda x: D2[x])#sorted(D2.items(), key=lambda kv: kv[1]) #sorted_names = sorted(scores, key=lambda x: scores[x]) for k in sorted_names:
    print(items, n)
    rank=1
    prescore=float(0)
    for i in range(n):
        cre2,score2=items[i]
        if score2 < prescore:
            rank=rank+1
        else:
            rank=rank
        if name not in D:
            D[name]=[(cre2, rank)]
        else:
            D[name].append((cre2, rank))
        prescore=score2


RF_dict={}
SVM_dict={}
#GB_dict={}
for dir in os.listdir(start_dir):
    if dir.startswith("run"):
        new_dir= str(start_dir)+'/'+str(dir)+'/'
        for file in os.listdir(new_dir):
            if file.endswith("imp") and file.startswith("Col"):
                print(file, "adding data to dictionary")
                name_list= file.strip().split('.fa')
                typex= name_list[-1]
                name= name_list[0]
                inp= open(new_dir+"/"+ file, 'r')
                if 'RF' in typex:
                    get_dicts(inp, name, RF_dict)
                elif 'SVM' in typex:
                    get_dicts(inp, name, SVM_dict)
                #elif 'GB' in typex:
                    #get_dicts(inp, name, GB_dict)
                else:
                    print(typex, "type doesn't have rank")
                inp.close()
                
#print(RF_dict, SVM_dict, "rank dictionaries")

def get_output(D, alg):
    for impfile in D:
        data_list= D[impfile]
        Dtmp={}
        print(impfile, alg, "getting average rank")
        for item in data_list:
            cre= item[0]
            rank= item[1]
            if cre not in Dtmp:
                Dtmp[cre]=[rank]
            else:
                Dtmp[cre].append(rank)
        #print(Dtmp)
        print(impfile, alg, "writing output")
        output= open(str(impfile)+"_imp_avgrank_"+str(alg)+'.txt', 'w')
        output2= open(str(impfile)+"_imp_allranks_"+str(alg)+'.txt', 'w')
        output.write('pCRE\taverage_rank\n')
        output2.write('pCRE\taverage_rank\tranks\n')
        for cre in Dtmp:
            ranklist= Dtmp[cre]
            avgrank= numpy.average(ranklist)
            output.write('%s\t%s\n' %(cre, avgrank))
            output2.write('%s\t%s\t' %(cre, avgrank))
            for r in ranklist:
                output2.write('%s\t' %(r))
            output2.write('\n')
        output.close()

get_output(RF_dict, 'RF')
get_output(SVM_dict, 'SVM')    