import pandas as pd
import numpy as np

known_data = [['A', 1, 4],['A', 2, 3],['A', 3, 2],['B', 8, 12],['B', 9, 13],['B', 10, 13]]
known = pd.DataFrame(known_data,columns = ['class','a1','a2'])
      
new = pd.DataFrame({'a1':[6,8],'a2':[5,9]})        
closest = pd.DataFrame()

for i in new.index:
    
    known['a1_0_'+str(i)] = known['a1'] - new['a1'][i]
    known['a2_0_'+str(i)] = known['a2'] - new['a2'][i]

    known['mag_1_'+str(i)] = np.sqrt(known['a1_0_'+str(i)]**2 + known['a2_0_'+str(i)]**2)

    k = 5
    smallest = known.nsmallest(k, 'mag_1_'+str(i))['class']
    smallest.reset_index(drop = True, inplace = True)
    closest = closest.append(smallest, ignore_index = True)
    
closest = closest.transpose()
vote = closest.apply(pd.value_counts) / k

print(vote)

    


