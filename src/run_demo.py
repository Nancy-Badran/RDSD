"""Reproducible synthetic BN experiment for a methodology poster."""
import json,csv,itertools,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
# Binary categories are deliberate simplifications, not empirical population facts.
SPEC=[
('AgeBand',['Younger','Older'],[], 'Demography'),
('Region',['Rural','Urban'],[], 'Demography'),
('EducationLevel',['Basic','Advanced'],['AgeBand','Region'],'Education'),
('StudyField',['General','Technical'],['EducationLevel','Region'],'Education'),
('EmploymentStatus',['NotEmployed','Employed'],['AgeBand','EducationLevel'],'Employment'),
('Occupation',['Other','Specialized'],['StudyField','EmploymentStatus'],'Employment'),
('Industry',['Other','Knowledge'],['Occupation','Region'],'Employment'),
('IncomeBand',['Lower','Higher'],['EducationLevel','Occupation'],'Finance'),
('MaritalStatus',['Single','Partnered'],['AgeBand','EmploymentStatus'],'Demography'),
('HouseholdSize',['Small','Large'],['MaritalStatus','AgeBand'],'Demography'),
('HousingType',['Apartment','House'],['HouseholdSize','IncomeBand'],'Finance'),
('HomeOwnership',['No','Yes'],['IncomeBand','AgeBand'],'Finance'),
('DebtLevel',['Lower','Higher'],['HousingType','HomeOwnership'],'Finance'),
('InternetUsage',['Lower','Higher'],['AgeBand','EducationLevel'],'Digital'),
('SocialMediaUsage',['Lower','Higher'],['InternetUsage','AgeBand'],'Digital'),
('OnlineShopping',['Lower','Higher'],['InternetUsage','IncomeBand'],'Digital'),
('DigitalLiteracy',['Lower','Higher'],['EducationLevel','InternetUsage'],'Digital'),
('HealthStatus',['FewerConditions','MoreConditions'],['AgeBand','EmploymentStatus'],'Health'),
('CareFrequency',['Lower','Higher'],['HealthStatus','IncomeBand'],'Health'),
('InsuranceCoverage',['Basic','Extended'],['EmploymentStatus','CareFrequency'],'Health')]
NAMES=[s[0] for s in SPEC]

def local_score(x,node,parents):
    q=2**len(parents);keys=np.zeros(len(x),dtype=int)
    for p in parents:keys=keys*2+x[:,p]
    c=np.bincount(keys*2+x[:,node],minlength=q*2).reshape(q,2)
    total=c.sum(axis=1,keepdims=True)
    mask=c>0
    ratios=np.divide(c,total,out=np.ones(c.shape,dtype=float),where=total>0)
    ll=float((c[mask]*np.log(ratios[mask])).sum())
    return ll-.5*q*np.log(len(x)),c

def learn(x):
    parents={};cpts={}
    for j,name in enumerate(NAMES):
        best=(-math.inf,None,None)
        for k in range(min(j,2)+1):
            for ps in itertools.combinations(range(j),k):
                score,c=local_score(x,j,ps)
                if score>best[0]:best=(score,ps,c)
        ps=best[1];counts=best[2];parents[name]=[NAMES[p] for p in ps]
        probs=(counts+1)/(counts.sum(axis=1,keepdims=True)+2)
        cpts[name]=[{'given':{NAMES[p]:SPEC[p][1][v] for p,v in zip(ps,values)},'probabilities':probs[i].tolist()} for i,values in enumerate(itertools.product([0,1],repeat=len(ps)))]
    return {'order':NAMES,'parents':parents,'cpts':cpts}

def nll(x,model):
    loss=0
    for j,name in enumerate(NAMES):
        keys=np.zeros(len(x),dtype=int)
        for parent in model['parents'][name]:keys=keys*2+x[:,NAMES.index(parent)]
        probs=np.array([r['probabilities'] for r in model['cpts'][name]])
        loss-=np.log(probs[keys,x[:,j]]).sum()
    return float(loss/len(x))

def generate():
    rng=np.random.default_rng(42);x=np.zeros((10000,20),dtype=int);reference={'order':NAMES,'parents':{},'cpts':{}}
    for j,(name,states,parents,group) in enumerate(SPEC):
        indices=[NAMES.index(p) for p in parents];reference['parents'][name]=parents
        # Explicit synthetic effects; a few negative signs avoid uniform positive associations.
        signs=[-1 if (j+k)%4==0 else 1 for k in range(len(parents))]
        bias=-.25 if parents else [-.2,.3][j]
        rows=[]
        for values in itertools.product([0,1],repeat=len(parents)):
            logit=bias+sum(sign*1.9*(2*v-1) for sign,v in zip(signs,values))
            p=1/(1+math.exp(-logit))
            rows.append({'given':dict(zip(parents,[SPEC[i][1][v] for i,v in zip(indices,values)])),'probabilities':[1-p,p]})
        reference['cpts'][name]=rows
        logit=np.full(len(x),bias)
        for idx,sign in zip(indices,signs):logit+=sign*1.9*(2*x[:,idx]-1)
        x[:,j]=(rng.random(len(x))<1/(1+np.exp(-logit))).astype(int)
    return x,reference

def figure(model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    # Five columns of four nodes; the title explicitly marks synthetic learned structure.
    pos={name:(j//4,3-j%4) for j,name in enumerate(NAMES)}
    colors={'Demography':'#71cbc5','Education':'#80b8e4','Employment':'#a6c982','Finance':'#f1bf6b','Digital':'#b39bdd','Health':'#e69baa'}
    fig,ax=plt.subplots(figsize=(20,11));ax.set_xlim(-.55,4.65);ax.set_ylim(-.85,3.8);ax.axis('off')
    for child in NAMES:
        for parent in model['parents'][child]:
            a,b=pos[parent],pos[child]
            curve=.12 if a[0]!=b[0] else .20
            arrow=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,connectionstyle=f'arc3,rad={curve}',shrinkA=22,shrinkB=23,color='#657d91',alpha=.65,lw=1.25,zorder=1)
            ax.add_patch(arrow)
    sensitive={'IncomeBand','DebtLevel','HealthStatus'}
    for name,states,parents,group in SPEC:
        xx,yy=pos[name];ax.scatter([xx],[yy],s=1700,c=[colors[group]],edgecolors='#21394b' if name in sensitive else 'white',linewidths=2.8,zorder=3)
        label=''.join(' '+c if c.isupper() else c for c in name).strip()
        ax.text(xx,yy-.31,label,ha='center',va='top',fontsize=11,fontweight='medium',bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':2},zorder=4)
    fig.suptitle('Synthetic identity ecosystem — learned Bayesian network',fontsize=23,fontweight='bold',y=.96)
    ax.set_title('10,000 profiles · 20 binary attributes · Ordered BIC structure learning',fontsize=14,pad=18,color='#52677b')
    handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=g) for g,c in colors.items()]
    fig.legend(handles=handles,loc='lower center',ncol=6,bbox_to_anchor=(.5,.065),frameon=False)
    fig.text(.5,.025,'Synthetic demonstration; arrows are statistical dependencies, not causal claims. Dark borders: scenario-sensitive targets. Uniform node size.',ha='center',fontsize=11,color='#52677b')
    fig.subplots_adjust(top=.84,bottom=.16,left=.04,right=.96)
    for ext in ['svg','png']:fig.savefig(ROOT/f'output/poster_graph.{ext}',dpi=250,facecolor='white')
    plt.close(fig)

def main():
    for folder in ['config','data','output']:(ROOT/folder).mkdir(exist_ok=True)
    x,reference=generate();attributes={name:{'states':states,'domain':group,'sensitive_target':name in {'IncomeBand','DebtLevel','HealthStatus'}} for name,states,ps,group in SPEC}
    (ROOT/'config/attributes.json').write_text(json.dumps({'attributes':attributes},indent=2))
    (ROOT/'config/reference_bn.json').write_text(json.dumps(reference,indent=2))
    with (ROOT/'data/profiles.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['ProfileID']+NAMES)
        w.writerows([f'P{i+1:06d}']+[SPEC[j][1][v] for j,v in enumerate(row)] for i,row in enumerate(x))
    indices=np.random.default_rng(43).permutation(len(x));train=x[indices[:8000]];test=x[indices[8000:]]
    model=learn(train)
    baseline={'order':NAMES,'parents':{n:[] for n in NAMES},'cpts':{}}
    for j,n in enumerate(NAMES):
        c=np.bincount(train[:,j],minlength=2);baseline['cpts'][n]=[{'given':{},'probabilities':((c+1)/(len(train)+2)).tolist()}]
    truth={(p,n) for n in NAMES for p in reference['parents'][n]};edges={(p,n) for n in NAMES for p in model['parents'][n]};tp=len(truth&edges)
    report={'synthetic':True,'profile_count':10000,'attribute_count':20,'train_count':8000,'test_count':2000,'generation_seed':42,'split_seed':43,'max_parents':2,'order_constraint':NAMES,'reference_edges':len(truth),'learned_edges':len(edges),'edge_precision':tp/len(edges) if edges else 0,'edge_recall':tp/len(truth),'test_mean_nll_nats':nll(test,model),'independent_baseline_test_mean_nll_nats':nll(test,baseline),'limits':'One synthetic scenario and one split; expert ordering supplied. Not empirical validation or causal discovery.'}
    (ROOT/'output/learned_bn.json').write_text(json.dumps(model,indent=2));(ROOT/'output/validation.json').write_text(json.dumps(report,indent=2))
    template=(ROOT/'src/ecosystem_template.html').read_text()
    # A taller view and a circular layout accommodate twenty draggable nodes.
    template=template.replace('height:620px','height:850px').replace('viewBox="0 0 900 620"','viewBox="0 0 1200 950"')
    start=template.index('function layout(){');end=template.index('\nlayout();',start)
    template=template[:start]+"function layout(){model.order.forEach((n,i)=>{const a=2*Math.PI*i/model.order.length-Math.PI/2;positions[n]={x:600+460*Math.cos(a),y:475+370*Math.sin(a)}});}"+template[end:]
    payload=json.dumps({'model':model,'attributes':attributes}).replace('<','\\u003c')
    (ROOT/'output/identity_ecosystem.html').write_text(template.replace('__MODEL_DATA__',payload),encoding='utf-8')
    figure(model);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
