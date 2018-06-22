import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import atecml.data 
import os


data_path = atecml.data.data_path
SAVE_PATH = atecml.data.system_root_path + 'feature_analytics/'


FEATURE_DESCRIPTION = data_path + 'fa_description'
output_dpi = 300


def feature_description(mode="train",groupby="date"):
    if (groupby == "date"):
        group = ['date']
    else:
        group = ['date','label']
    if (mode == "train"):
        df = atecml.data.load_train()
        PICKLE_PATH = FEATURE_DESCRIPTION + '.train' + groupby +'.dat'
    else:
        df = atecml.data.load_test()
        PICKLE_PATH = FEATURE_DESCRIPTION + '.test'+ groupby +'.dat'
        
    if (os.path.exists(PICKLE_PATH)):
        group_df = pd.read_pickle(PICKLE_PATH)
    else:
        group_df = df.groupby(group).describe()
        group_df.to_pickle(PICKLE_PATH)
    return group_df
      


def summary_traindf(train_df,describe_df,feature_name):
    """
    Matplotlib说实话不方便也懒得学，更多的时候还是用plotly和bokeh
    但是bokeh采用HTML渲染，图多了后浏览器特别卡
    这次只是时间紧，懒得多研究，就这样ugly的弄一起好了
    """
    chart_name = 'Feature_Analytics__TRAIN_DF_Summary__'+feature_name
    export_filename = SAVE_PATH + chart_name +'.png'

    f = plt.figure(figsize=(20,4*7))
    df= train_df[[feature_name,'label']].dropna()

    ax0 = f.add_subplot(721) #总行数、总列数、子图位置
    ax0.set_title(chart_name)
    ax0.hist(df[feature_name][df['label']==0],bins=30,color='#b2d235',alpha = 0.5)
    ax0.hist(df[feature_name][df['label']==1],bins=30,color='#002299',alpha = 0.5)
    ax0.set_yscale('log')
    ax0.set_xlabel('histogram of: ' + str(feature_name))
    ax0.set_autoscaley_on(False)

    ax1 = f.add_subplot(722) 
    ax1.set_title(chart_name)
    ax1.hist(df[feature_name][df['label']==0],bins=30,color='#b2d235',alpha = 0.5, density=True)
    ax1.hist(df[feature_name][df['label']==1],bins=30,color='#002299',alpha = 0.5, density=True)
    ax1.set_yscale('log')
    ax1.set_xlabel('[Density]histogram of: ' + str(feature_name))
    ax1.set_autoscaley_on(False)

    ax2 = f.add_subplot(712)
    ax2.set_title('normal[count]')
    ax2_value = describe_df[feature_name]['count'].unstack()[0]
    ax2.plot(ax2_value.index,ax2_value,'g')
    ax2.set_ylabel( feature_name+'    [ stats_by: count]')

    ax3 = f.add_subplot(713)
    ax3.set_title('fraud[count]')
    ax3_value = describe_df[feature_name]['count'].unstack()[1]
    ax3.plot(ax3_value.index,ax3_value,'r')
    ax3.set_ylabel(feature_name+'    [ stats_by: count]')

    ax4 = f.add_subplot(714)
    ax4.set_title('normal[quad]')
    ax4_mean = describe_df[feature_name]['50%'].unstack()[0]
    ax4_low = describe_df[feature_name]['25%'].unstack()[0]
    ax4_high = describe_df[feature_name]['75%'].unstack()[0]
    ax4.plot(ax4_mean.index,ax4_mean,'k')
    ax4.fill_between(ax4_mean.index, ax4_low, ax4_high, color='g', alpha=0.2)
    ax4.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')

    ax5 = f.add_subplot(715)
    ax5.set_title('fruad[quad]')
    ax5_mean = describe_df[feature_name]['50%'].unstack()[1]
    ax5_low = describe_df[feature_name]['25%'].unstack()[1]
    ax5_high = describe_df[feature_name]['75%'].unstack()[1]
    ax5.plot(ax5_mean.index,ax5_mean,'k')
    ax5.fill_between(ax5_mean.index, ax5_low, ax5_high, color='r', alpha=0.2)
    ax5.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')

    ax6 = f.add_subplot(716)
    ax6.set_title('normal[mean/std]')
    ax6_mean = describe_df[feature_name]['mean'].unstack()[0]
    ax6_std = describe_df[feature_name]['std'].unstack()[0]
    ax6.plot(ax6_mean.index,ax6_mean,'k')
    ax6.fill_between(ax6_mean.index, ax6_mean-2*ax6_std, ax6_mean+2*ax6_std, color='g', alpha=0.2)
    ax6.set_ylabel(feature_name+'    [ stats_by: mean/std]')

    ax7 = f.add_subplot(717)
    ax7.set_title('fruad[mean/std]')
    ax7_mean = describe_df[feature_name]['mean'].unstack()[1]
    ax7_std = describe_df[feature_name]['std'].unstack()[1]
    ax7.plot(ax7_mean.index,ax7_mean,'k')
    ax7.fill_between(ax7_mean.index, ax7_mean-2*ax7_std, ax7_mean+2*ax7_std, color='r', alpha=0.2)
    ax7.set_ylabel(feature_name+'    [ stats_by: mean/std]')
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
    plt.margins(0,0)  
    plt.savefig(export_filename,dpi=100, pad_inches = 0)
    plt.show()

def summary_testdf(test_df,describe_df,feature_name):
    """
    Matplotlib说实话不方便也懒得学，更多的时候还是用plotly和bokeh
    但是bokeh采用HTML渲染，图多了后浏览器特别卡
    这次只是时间紧，懒得多研究，就这样ugly的弄一起好了
    """
    chart_name = 'Feature_Analytics__TEST_DF_Summary__'+feature_name
    export_filename = SAVE_PATH + chart_name +'.png'

    f = plt.figure(figsize=(20,5*4))
    df= test_df[feature_name].dropna()

    ax0 = f.add_subplot(421) 
    ax0.set_title(chart_name)
    ax0.hist(df,bins=30,color='#b2d235',alpha = 0.5)
    ax0.set_yscale('log')
    ax0.set_xlabel('histogram of: ' + str(feature_name))
    ax0.set_autoscaley_on(False)

    ax1 = f.add_subplot(422) 
    ax1.set_title(chart_name)
    ax1.hist(df,bins=30,color='#b2d235',alpha = 0.5, density=True)
    ax1.set_yscale('log')
    ax1.set_xlabel('[Density]histogram of: ' + str(feature_name))
    ax1.set_autoscaley_on(False)

    ax2 = f.add_subplot(412)
    ax2.set_title('count')
    ax2_value = describe_df[feature_name]['count']
    ax2.plot(ax2_value.index,ax2_value,'g')
    ax2.set_ylabel( feature_name+'    [ stats_by: count]')

    ax4 = f.add_subplot(413)
    ax4.set_title('quad')
    ax4_mean = describe_df[feature_name]['50%']
    ax4_low = describe_df[feature_name]['25%']
    ax4_high = describe_df[feature_name]['75%']
    ax4.plot(ax4_mean.index,ax4_mean,'k')
    ax4.fill_between(ax4_mean.index, ax4_low, ax4_high, color='g', alpha=0.2)
    ax4.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')


    ax6 = f.add_subplot(414)
    ax6.set_title('mean/std')
    ax6_mean = describe_df[feature_name]['mean']
    ax6_std = describe_df[feature_name]['std']
    ax6.plot(ax6_mean.index,ax6_mean,'k')
    ax6.fill_between(ax6_mean.index, ax6_mean-2*ax6_std, ax6_mean+2*ax6_std, color='g', alpha=0.2)
    ax6.set_ylabel(feature_name+'    [ stats_by: mean/std]')

    plt.margins(0,0)  
    plt.savefig(export_filename,dpi=100, pad_inches = 0)
    plt.show()


    
def compare_train_test(train_df,train_describe,test_df,test_describe,feature_name,savefig=True):
    """
    Matplotlib说实话不方便也懒得学，更多的时候还是用plotly和bokeh
    但是bokeh采用HTML渲染，图多了后浏览器特别卡
    这次只是时间紧，懒得多研究，就这样ugly的弄一起好了
    """
    chart_name = 'Feature_Analytics__TRAIN_TEST_Compare__'+feature_name
    export_filename = SAVE_PATH + chart_name +'.png'
    

    f = plt.figure(figsize=(37,4*7))
    df= train_df[[feature_name,'label']].dropna()
    df1= test_df[[feature_name]].dropna()
    
    describe_df = train_describe
    predict_discribe =test_describe

    ax0 = f.add_subplot(741) 
    ax0.set_title(chart_name+"[Train]")
    ax0.hist(df[feature_name][df['label']==0],bins=30,color='#b2d235',alpha = 0.5)
    ax0.hist(df[feature_name][df['label']==1],bins=30,color='#002299',alpha = 0.5)
    ax0.set_yscale('log')
    ax0.set_xlabel('histogram of: ' + str(feature_name))
    ax0.set_autoscaley_on(False)

    ax1 = f.add_subplot(742) 
    ax1.hist(df[feature_name][df['label']==0],bins=30,color='#b2d235',alpha = 0.5, density=True)
    ax1.hist(df[feature_name][df['label']==1],bins=30,color='#002299',alpha = 0.5, density=True)
    ax1.set_yscale('log')
    ax1.set_xlabel('[Density]histogram of: ' + str(feature_name))
    ax1.set_autoscaley_on(False)


    bx0 = f.add_subplot(743) 
    bx0.set_title(chart_name+"[Test]")
    bx0.hist(df1[feature_name],bins=30,color='#b2d235',alpha = 0.5)
    bx0.set_yscale('log')
    bx0.set_xlabel('histogram of: ' + str(feature_name))
    bx0.set_autoscaley_on(False)

    bx1 = f.add_subplot(744) 
    bx1.hist(df1[feature_name],bins=30,color='#b2d235',alpha = 0.5, density=True)
    bx1.set_yscale('log')
    bx1.set_xlabel('[Density]histogram of: ' + str(feature_name))
    bx1.set_autoscaley_on(False)
    
    
    
    ax2 = f.add_subplot(723)
    ax2.set_title('normal[count]')
    ax2_value = describe_df[feature_name]['count'].unstack()[0]
    ax2.plot(ax2_value.index,ax2_value,'g')
    ax2.set_ylabel( feature_name+'    [ stats_by: count]')

    bx2 = f.add_subplot(724)
    bx2.set_title('normal[count]')
    bx2_value = predict_discribe[feature_name]['count']
    bx2.plot(bx2_value.index,bx2_value,'g')
    bx2.set_ylabel( feature_name+'    [ stats_by: count]')    
    
    ax3 = f.add_subplot(725)
    ax3.set_title('fraud[count]')
    ax3_value = describe_df[feature_name]['count'].unstack()[1]
    ax3.plot(ax3_value.index,ax3_value,'r')
    ax3.set_ylabel(feature_name+'    [ stats_by: count]')
   
    
    ax4 = f.add_subplot(727)
    ax4.set_title('normal[quad]')
    ax4_mean = describe_df[feature_name]['50%'].unstack()[0]
    ax4_low = describe_df[feature_name]['25%'].unstack()[0]
    ax4_high = describe_df[feature_name]['75%'].unstack()[0]
    ax4.plot(ax4_mean.index,ax4_mean,'k')
    ax4.fill_between(ax4_mean.index, ax4_low, ax4_high, color='g', alpha=0.2)
    ax4.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')

    bx4 = f.add_subplot(728)
    bx4.set_title('normal[quad]')
    bx4_mean = predict_discribe[feature_name]['50%']
    bx4_low = predict_discribe[feature_name]['25%']
    bx4_high = predict_discribe[feature_name]['75%']
    bx4.plot(bx4_mean.index,bx4_mean,'k')
    bx4.fill_between(bx4_mean.index, bx4_low, bx4_high, color='b', alpha=0.2)
    bx4.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')   
    
    ax5 = f.add_subplot(729)
    ax5.set_title('fruad[quad]')
    ax5_mean = describe_df[feature_name]['50%'].unstack()[1]
    ax5_low = describe_df[feature_name]['25%'].unstack()[1]
    ax5_high = describe_df[feature_name]['75%'].unstack()[1]
    ax5.plot(ax5_mean.index,ax5_mean,'k')
    ax5.fill_between(ax5_mean.index, ax5_low, ax5_high, color='r', alpha=0.2)
    ax5.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')
    
    ax6 = f.add_subplot(7,2,11)
    ax6.set_title('normal[mean/std]')
    ax6_mean = describe_df[feature_name]['mean'].unstack()[0]
    ax6_std = describe_df[feature_name]['std'].unstack()[0]
    ax6.plot(ax6_mean.index,ax6_mean,'k')
    ax6.fill_between(ax6_mean.index, ax6_mean-2*ax6_std, ax6_mean+2*ax6_std, color='g', alpha=0.2)
    ax6.set_ylabel(feature_name+'    [ stats_by: mean/std]')
    
    bx6 = f.add_subplot(7,2,12)
    bx6.set_title('normal[mean/std]')
    bx6_mean = predict_discribe[feature_name]['mean']
    bx6_std = predict_discribe[feature_name]['std']
    bx6.plot(bx6_mean.index,bx6_mean,'k')
    bx6.fill_between(bx6_mean.index, bx6_mean-2*bx6_std, bx6_mean+2*bx6_std, color='b', alpha=0.2)
    bx6.set_ylabel(feature_name+'    [ stats_by: mean/std]')    

    ax7 = f.add_subplot(7,2,13)
    ax7.set_title('fruad[mean/std]')
    ax7_mean = describe_df[feature_name]['mean'].unstack()[1]
    ax7_std = describe_df[feature_name]['std'].unstack()[1]
    ax7.plot(ax7_mean.index,ax7_mean,'k')
    ax7.fill_between(ax7_mean.index, ax7_mean-2*ax7_std, ax7_mean+2*ax7_std, color='r', alpha=0.2)
    ax7.set_ylabel(feature_name+'    [ stats_by: mean/std]')
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
    plt.margins(0,0)  
    plt.tight_layout()
    if (savefig):
        plt.savefig(export_filename,dpi=output_dpi, pad_inches = 0)
    plt.show()    
    

def compare_train(data_set,describe_df,data_set1,describe_df1,feature_name,savefig=True):
    """
    Matplotlib说实话不方便也懒得学，更多的时候还是用plotly和bokeh
    但是bokeh采用HTML渲染，图多了后浏览器特别卡
    这次只是时间紧，懒得多研究，就这样ugly的弄一起好了
    """
    chart_name = 'Feature_Analytics__Compare_Trained_AB__'+feature_name
    export_filename = SAVE_PATH + chart_name +'.png'


    f = plt.figure(figsize=(37,4*7))
    df= data_set[[feature_name,'label']].dropna()
    df1= data_set1[[feature_name,'label']].dropna()
    

    ax0 = f.add_subplot(741) #总行数、总列数、子图位置
    ax0.set_title(chart_name+"[Train]")
    ax0.hist(df[feature_name][df['label']==0],bins=30,color='#b2d235',alpha = 0.5)
    ax0.hist(df[feature_name][df['label']==1],bins=30,color='#002299',alpha = 0.5)
    ax0.set_yscale('log')
    ax0.set_xlabel('histogram of: ' + str(feature_name))
    ax0.set_autoscaley_on(False)

    ax1 = f.add_subplot(742) #总行数、总列数、子图位置
    ax1.hist(df[feature_name][df['label']==0],bins=30,color='#b2d235',alpha = 0.5, density=True)
    ax1.hist(df[feature_name][df['label']==1],bins=30,color='#002299',alpha = 0.5, density=True)
    ax1.set_yscale('log')
    ax1.set_xlabel('[Density]histogram of: ' + str(feature_name))
    ax1.set_autoscaley_on(False)
      
    ax2 = f.add_subplot(723)
    ax2.set_title('normal[count]')
    ax2_value = describe_df[feature_name]['count'].unstack()[0]
    ax2.plot(ax2_value.index,ax2_value,'g')
    ax2.set_ylabel( feature_name+'    [ stats_by: count]')
    
    ax3 = f.add_subplot(725)
    ax3.set_title('fraud[count]')
    ax3_value = describe_df[feature_name]['count'].unstack()[1]
    ax3.plot(ax3_value.index,ax3_value,'r')
    ax3.set_ylabel(feature_name+'    [ stats_by: count]')    
  
    ax4 = f.add_subplot(727)
    ax4.set_title('normal[quad]')
    ax4_mean = describe_df[feature_name]['50%'].unstack()[0]
    ax4_low = describe_df[feature_name]['25%'].unstack()[0]
    ax4_high = describe_df[feature_name]['75%'].unstack()[0]
    ax4.plot(ax4_mean.index,ax4_mean,'k')
    ax4.fill_between(ax4_mean.index, ax4_low, ax4_high, color='g', alpha=0.2)
    ax4.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')
   
    
    ax5 = f.add_subplot(729)
    ax5.set_title('fruad[quad]')
    ax5_mean = describe_df[feature_name]['50%'].unstack()[1]
    ax5_low = describe_df[feature_name]['25%'].unstack()[1]
    ax5_high = describe_df[feature_name]['75%'].unstack()[1]
    ax5.plot(ax5_mean.index,ax5_mean,'k')
    ax5.fill_between(ax5_mean.index, ax5_low, ax5_high, color='r', alpha=0.2)
    ax5.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')
    
    ax6 = f.add_subplot(7,2,11)
    ax6.set_title('normal[mean/std]')
    ax6_mean = describe_df[feature_name]['mean'].unstack()[0]
    ax6_std = describe_df[feature_name]['std'].unstack()[0]
    ax6.plot(ax6_mean.index,ax6_mean,'k')
    ax6.fill_between(ax6_mean.index, ax6_mean-2*ax6_std, ax6_mean+2*ax6_std, color='g', alpha=0.2)
    ax6.set_ylabel(feature_name+'    [ stats_by: mean/std]')

    ax7 = f.add_subplot(7,2,13)
    ax7.set_title('fruad[mean/std]')
    ax7_mean = describe_df[feature_name]['mean'].unstack()[1]
    ax7_std = describe_df[feature_name]['std'].unstack()[1]
    ax7.plot(ax7_mean.index,ax7_mean,'k')
    ax7.fill_between(ax7_mean.index, ax7_mean-2*ax7_std, ax7_mean+2*ax7_std, color='r', alpha=0.2)
    ax7.set_ylabel(feature_name+'    [ stats_by: mean/std]')
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
    
    bx0 = f.add_subplot(743)
    bx0.set_title(chart_name+"[Train]")
    bx0.hist(df1[feature_name][df1['label']==0],bins=30,color='#b2d235',alpha = 0.5)
    bx0.hist(df1[feature_name][df1['label']==1],bins=30,color='#002299',alpha = 0.5)
    bx0.set_yscale('log')
    bx0.set_xlabel('histogram of: ' + str(feature_name))
    bx0.set_autoscaley_on(False)

    bx1 = f.add_subplot(744)
    bx1.hist(df1[feature_name][df1['label']==0],bins=30,color='#b2d235',alpha = 0.5, density=True)
    bx1.hist(df1[feature_name][df1['label']==1],bins=30,color='#002299',alpha = 0.5, density=True)
    bx1.set_yscale('log')
    bx1.set_xlabel('[Density]histogram of: ' + str(feature_name))
    bx1.set_autoscaley_on(False)
    
    bx2 = f.add_subplot(724)
    bx2.set_title('normal[count]')
    bx2_value = describe_df1[feature_name]['count'].unstack()[0]
    bx2.plot(bx2_value.index,bx2_value,'g')
    bx2.set_ylabel( feature_name+'    [ stats_by: count]')
    
    bx3 = f.add_subplot(726)
    bx3.set_title('fraud[count]')
    bx3_value = describe_df1[feature_name]['count'].unstack()[1]
    bx3.plot(bx3_value.index,bx3_value,'r')
    bx3.set_ylabel(feature_name+'    [ stats_by: count]')        

    bx4 = f.add_subplot(728)
    bx4.set_title('normal[quad]')
    bx4_mean = describe_df1[feature_name]['50%'].unstack()[0]
    bx4_low = describe_df1[feature_name]['25%'].unstack()[0]
    bx4_high = describe_df1[feature_name]['75%'].unstack()[0]
    bx4.plot(bx4_mean.index,bx4_mean,'k')
    bx4.fill_between(bx4_mean.index, bx4_low, bx4_high, color='g', alpha=0.2)
    bx4.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')
    
    
    bx5 = f.add_subplot(7,2,10)
    bx5.set_title('fruad[quad]')
    bx5_mean = describe_df1[feature_name]['50%'].unstack()[1]
    bx5_low = describe_df1[feature_name]['25%'].unstack()[1]
    bx5_high = describe_df1[feature_name]['75%'].unstack()[1]
    bx5.plot(bx5_mean.index,bx5_mean,'k')
    bx5.fill_between(bx5_mean.index, bx5_low, bx5_high, color='r', alpha=0.2)
    bx5.set_ylabel( feature_name+'    [ stats_by: 25%-50%-75% ]')
    
    bx6 = f.add_subplot(7,2,12)
    bx6.set_title('normal[mean/std]')
    bx6_mean = describe_df1[feature_name]['mean'].unstack()[0]
    bx6_std = describe_df1[feature_name]['std'].unstack()[0]
    bx6.plot(bx6_mean.index,bx6_mean,'k')
    bx6.fill_between(bx6_mean.index, bx6_mean-2*bx6_std, bx6_mean+2*bx6_std, color='g', alpha=0.2)
    bx6.set_ylabel(feature_name+'    [ stats_by: mean/std]')

    bx7 = f.add_subplot(7,2,14)
    bx7.set_title('fruad[mean/std]')
    bx7_mean = describe_df1[feature_name]['mean'].unstack()[1]
    bx7_std = describe_df1[feature_name]['std'].unstack()[1]
    bx7.plot(bx7_mean.index,bx7_mean,'k')
    bx7.fill_between(bx7_mean.index, bx7_mean-2*bx7_std, bx7_mean+2*bx7_std, color='r', alpha=0.2)
    bx7.set_ylabel(feature_name+'    [ stats_by: mean/std]')
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)  
 
    plt.margins(0,0)  
    plt.tight_layout()
    if (savefig):
        plt.savefig(export_filename,dpi=output_dpi, pad_inches = 0)
    plt.show()    