import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('211.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        #y.append(float(row[2]))
        if row[2][:1] == '-' :
            if row[2][1:len(row[2])] == '' :
                #print(float(row[0]),'error')
                y.append(0)
            else :
                curr = float(row[2][1:len(row[2])])
                curr = float(-curr)
                #print(curr)
                y.append(curr)
        else :
            y.append(float(row[2]))
            #print('test')

plt.plot(x,y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('PPG')
plt.legend()
plt.show()
