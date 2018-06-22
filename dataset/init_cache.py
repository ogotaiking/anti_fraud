import atecml.data

print('Building Training Data Cache........')
x = atecml.data.load_train()
print('Done....... Records:',len(x))

print('Building Testing Data Cache........')
y = atecml.data.load_test()
print('Done....... Records:',len(y))
