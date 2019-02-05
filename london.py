import pandas as pd
import numpy as np

# Clean the dataset

oysteroriginal = pd.read_csv('Nov09JnyExport.csv')

oyster = oysteroriginal

oyster = oyster.drop(['downo'], axis=1)
oyster = oyster.drop(['JNYTYP'], axis=1)
oyster = oyster.drop(['FFare'], axis=1)
oyster = oyster.drop(['DFare'], axis=1)
oyster = oyster.drop(['RouteID'], axis=1)
oyster = oyster.drop(['FinalProduct'], axis=1)
oyster = oyster.drop(['DailyCapping'], axis=1)
oyster = oyster.drop(['EntTimeHHMM'], axis=1)
oyster = oyster.drop(['EXTimeHHMM'], axis=1)

oyster = oyster[oyster.StartStn!='Unstarted']
oyster = oyster[oyster.EndStation!='Not Applicable']
oyster = oyster[oyster.ExTime!=0]

oyster = oyster.reset_index(drop=True)

oyster = oyster[oyster.daytype!='Sat']
oyster = oyster[oyster.daytype!='Sun']

start = 280
end = 1380
interval = 10

oyster['Entinterval'] = pd.Series(0, index=oyster.index)
oyster['Extinterval'] = pd.Series(0, index=oyster.index)

for day in set(oyster.daytype.unique()):
    oyster.loc[(oyster.daytype==day) & (oyster.EntTime<=start), 'Entinterval'] = 1
    oyster.loc[(oyster.daytype==day) & (oyster.ExTime<=start), 'Extinterval'] = 1

intervalname = 1

while start<end:
    intervalname = intervalname +1
    for day in set(oyster.daytype.unique()):

        oyster.loc[(oyster.daytype==day) & (oyster.EntTime>start) & (oyster.EntTime<=start+interval),'Entinterval'] = intervalname
        oyster.loc[(oyster.daytype==day) & (oyster.ExTime>start) & (oyster.ExTime<=start+interval),'Extinterval'] = intervalname        

    start=start+interval
        
for day in set(oyster.daytype.unique()):
    oyster.loc[(oyster.daytype==day) & (oyster.EntTime>start), 'Entinterval'] = intervalname+1
    oyster.loc[(oyster.daytype==day) & (oyster.ExTime>start), 'Extinterval'] = intervalname+1    



oyster = oyster.drop(['EntTime'], axis=1)
oyster = oyster.drop(['ExTime'], axis=1)
oyster = oyster.drop(['ZVPPT'], axis=1)


oyster = oyster[oyster.SubSystem=='LUL']

oyster = oyster.drop(['SubSystem'], axis=1)

oyster = oyster.reset_index(drop=True)

sument = max(oyster.loc[oyster.daytype=='Mon']['Entinterval'])
sumext = max(oyster.loc[oyster.daytype=='Mon']['Extinterval'])

nextent = max(oyster.loc[oyster.daytype=='Tue']['Entinterval'])
nextext = max(oyster.loc[oyster.daytype=='Tue']['Extinterval'])

oyster.loc[oyster.daytype== 'Tue', 'Entinterval']= oyster.loc[oyster.daytype== 'Tue', 'Entinterval']+sument
oyster.loc[oyster.daytype== 'Tue', 'Extinterval']= oyster.loc[oyster.daytype== 'Tue', 'Extinterval']+sumext

sument = sument+  nextent
sumext = sumext+  nextext

nextent = max(oyster.loc[oyster.daytype=='Wed']['Entinterval'])
nextext = max(oyster.loc[oyster.daytype=='Wed']['Extinterval'])

oyster.loc[oyster.daytype== 'Wed', 'Entinterval']= oyster.loc[oyster.daytype== 'Wed', 'Entinterval']+sument
oyster.loc[oyster.daytype== 'Wed', 'Extinterval']= oyster.loc[oyster.daytype== 'Wed', 'Extinterval']+sumext

sument = sument+  nextent
sumext = sumext+  nextext

nextent = max(oyster.loc[oyster.daytype=='Thu']['Entinterval'])
nextext = max(oyster.loc[oyster.daytype=='Thu']['Extinterval'])

oyster.loc[oyster.daytype== 'Thu', 'Entinterval']= oyster.loc[oyster.daytype== 'Thu', 'Entinterval']+sument
oyster.loc[oyster.daytype== 'Thu', 'Extinterval']= oyster.loc[oyster.daytype== 'Thu', 'Extinterval']+sumext

sument = sument+  nextent
sumext = sumext+  nextext

nextent = max(oyster.loc[oyster.daytype=='Fri']['Entinterval'])
nextext = max(oyster.loc[oyster.daytype=='Fri']['Extinterval'])

oyster.loc[oyster.daytype== 'Fri', 'Entinterval']= oyster.loc[oyster.daytype== 'Fri', 'Entinterval']+sument
oyster.loc[oyster.daytype== 'Fri', 'Extinterval']= oyster.loc[oyster.daytype== 'Fri', 'Extinterval']+sumext

oyster.loc[(oyster.EndStation == 'Kings Cross M'), 'EndStation']= "King's Cross St. Pancras"
oyster.loc[(oyster.EndStation== 'Kings Cross T'), 'EndStation']= "King's Cross St. Pancras"
oyster.loc[(oyster.StartStn=='Kings Cross M'),'StartStn']= "King's Cross St. Pancras"
oyster.loc[(oyster.StartStn== 'Kings Cross T'),'StartStn']= "King's Cross St. Pancras"

oyster.loc[(oyster.EndStation == "Shepherd's Bush Mkt"), 'EndStation']= "Shepherd's Bush"
oyster.loc[(oyster.EndStation== "Shepherd's Bush Und"), 'EndStation']= "Shepherd's Bush"
oyster.loc[(oyster.StartStn== "Shepherd's Bush Mkt"),'StartStn']= "Shepherd's Bush"
oyster.loc[(oyster.StartStn== "Shepherd's Bush Und"),'StartStn']= "Shepherd's Bush"

oyster.loc[(oyster.EndStation== "Canary Wharf E2"), 'EndStation']= "Canary Wharf"
oyster.loc[(oyster.StartStn== "Canary Wharf E2"),'StartStn']= "Canary Wharf"

oyster.loc[(oyster.EndStation == "Hammersmith M"), 'EndStation']= "Hammersmith"
oyster.loc[(oyster.EndStation== "Hammersmith D"), 'EndStation']= "Hammersmith"
oyster.loc[(oyster.StartStn== "Hammersmith M"),'StartStn']= "Hammersmith"
oyster.loc[(oyster.StartStn== "Hammersmith D"),'StartStn']= "Hammersmith"

oyster.loc[(oyster.EndStation== "Waterloo JLE"), 'EndStation']= "Waterloo"
oyster.loc[(oyster.StartStn== "Waterloo JLE"),'StartStn']= "Waterloo"

oyster.loc[(oyster.EndStation == ""), 'EndStation']= ""
oyster.loc[(oyster.EndStation== ""), 'EndStation']= ""
oyster.loc[(oyster.StartStn== ""),'StartStn']= ""
oyster.loc[(oyster.StartStn== ""),'StartStn']= ""

oyster = oyster[oyster.EndStation !='Heathrow Term 5']
oyster = oyster[oyster.StartStn !='Heathrow Term 5']

oyster = oyster[oyster.EndStation !='London City Airport']
oyster = oyster[oyster.StartStn !='London City Airport']

stationset = set(oyster.EndStation).union(set(oyster.StartStn))

maxent = np.max(oyster['Entinterval'])
minent = np.min(oyster['Entinterval'])
maxext = np.max(oyster['Extinterval'])
minext = np.min(oyster['Extinterval'])

maxtime= np.max(np.array([maxent,maxext]))
mintime= np.min(np.array([minent,minext]))

stations= (set(oyster['EndStation'].unique())).union(set(oyster['StartStn'].unique()))

londonunderground = pd.DataFrame(columns=['station', 'time', 'flow'])

i=0
for station in stations:
    for time in range(mintime,maxtime+1):
        instat = len(oyster.loc[(oyster.StartStn==station) & ((oyster.Entinterval==time))])
        outstat = len(oyster.loc[(oyster.StartStn==station) & ((oyster.Extinterval==time))])
        totstat = instat-outstat
        londonunderground.loc[i] = [station, time, totstat]
        i=i+1

londonunderground.to_csv('londonunderground10min.csv', sep='\t')

# Load the edges and the nodes of the graph

london2= open( "London2.pickle", "rb" )

nodes, edges = pickle.load(london2)
oystergraph=nx.Graph()
oystergraph.add_nodes_from(nodes)
oystergraph.add_edges_from(edges)

# Load the polished dataset

londonunderground = pd.read_csv('londonunderground10min.csv',sep='\t')

lanepart= nx.shortest_path(oystergraph, 'Upminster Bridge', 'Plaistow')

lanepartLondon = londonunderground.loc[londonunderground.station.isin(list(lanepart))]
lanepartLondon= lanepartLondon.reset_index(drop=True)
names=np.unique(np.array(lanepartLondon.station))

# Set the training set
lanepartLondontrain = lanepartLondon.loc[(lanepartLondon.time>=0) & (lanepartLondon.time<=224)].reset_index(drop=True)

lanepartLondontrainYT = np.zeros(len(np.unique(lanepartLondontrain.station)))

for t in range(1, 225):
    supp = np.zeros(len(np.unique(lanepartLondontrain.station)))
    i = 0
    for stat in lanepart:
        supp[i] = float(lanepartLondontrain.loc[(lanepartLondontrain.station == stat) & (lanepartLondontrain.time==t)].flow)
        i= i+1
    
    lanepartLondontrainYT = np.vstack((lanepartLondontrainYT, supp))

T= np.size(lanepartLondontrainYT,0)
#nlayers = np.size(lanepartLondontrainYT,1)*2
nlayers = np.size(lanepartLondontrainYT,1)+1
nstates = 3
nobservations = np.size(lanepartLondontrainYT,1)

trivialpart = list()

for i in range(0,nlayers):
    trivialpart.append(['H'+str(i)])



# partition choice
trivialpart = list()

for i in range(0,nlayers):
    trivialpart.append(['H'+str(i)])


############################
# EM algorithm

EMdict = {}

# initial conditions
initial0 = list()

P0 = list()

# Create different initial conditions

c0 = np.linspace(0.25,10,20)

sigma0 = np.linspace(0.5,10,20)

for i in np.linspace(0,19,20):
    mu0new = np.array([0+2.5*(i/100),1-5*(i/100),0+2.5*(i/100)], dtype=float)
    initial0.append(mu0new)
    
    Pnew = np.array([[0+2.5*(i/100), 1-5*(i/100), 0+2.5*(i/100)],
                     [0.5-2.5*(i/100), 5*(i/100), 0.5-2.5*(i/100)],
                     [0+2.5*(i/100), 1-5*(i/100), 0+2.5*(i/100)]], dtype=float)

    P0.append(Pnew)

for i in range(-5,5):
    mu0new = np.array([0.5-5*(i/100),0,0.5+5*(i/100)], dtype=float)
    initial0.append(mu0new)
    
    Pnew = np.array([[ .5-5*(i/100)-0.005, 0.01,  .5+5*(i/100)-0.005],
                     [0.5               , 0.01, 0.49              ],
                     [ .5+5*(i/100)-0.005, 0.01,  .5-5*(i/100)-0.005]], dtype=float)
    print(sum(sum(Pnew)))
    P0.append(Pnew)
    
m=0
statevalue= np.array([-1,0,1],dtype=float)

# Run the EM for different initial conditions

print('start EM')

for value in range(0,20):

    EMdict['cond'+str(value)] = EMalgorithmtoy.EMlondon(nlayers, lanepartLondontrainYT, initial0[value], P0[value], c0[value], sigma0[value], trivialpart, m, 100, statevalue)
    title = str('EMlondon-done')+str(value)
    f= open(title,"w+")
    f.close()



##########################################
# FILES

EMf = open("londonEMmultiple.pkl","wb")
pickle.dump(EMdict,EMf)
EMf.close()
