import tabulate as tabulate
import pandas as pd
import numpy as np
import bokeh.plotting
import bokeh.models
import bokeh.layouts
import bokeh.palettes

bokeh.plotting.output_notebook()


def pretty_print_df(df,columns=[]):
    if (len(columns) == 0):
       columns = df.columns
    return tabulate.tabulate(df,columns)

def bokeh_multi_line_chart(df,item_list,legend_list,title,width=1900,height=380,legend_location='bottom_left',x_axis='dateTime',x_axis_type='datetime',y_axis_type='auto',line_width=1.5,alpha=0.7):
    fig = bokeh.plotting.figure(width=width,height=height,x_axis_type=x_axis_type , y_axis_type=y_axis_type ,title=title)
    lines_counter = len (item_list)
    if (lines_counter <= 3):
        color_list=['#d25535','#35b2d2','#98d235']
    elif (lines_counter <=10):
        color_list=bokeh.palettes.Category10[10]
    else:
        color_list=bokeh.palettes.Category20[20]

    for idx in range(0,lines_counter):
        item = item_list[idx]
        label = legend_list[idx]
        fig.line(df[x_axis],df[item],color=color_list[idx],legend=label,line_width=line_width,alpha=alpha)
    fig.legend.location = legend_location
    fig.legend.label_text_font_size = "0.8em"
    return fig

def bokeh_hbar_chart(df,categories_col,value_col,title,color='#B2D235',width=400,height=300):
    categories = list(df[categories_col])
    categories.reverse()
    result_df = df[[categories_col,value_col]]
    source = bokeh.models.ColumnDataSource(result_df)    
    fig = bokeh.plotting.figure(title=title, y_range=bokeh.models.FactorRange(factors=categories), width=width,height=height)
    fig.hbar(left=0, y=categories_col,right=value_col, color=color, source=source,height=0.3)
    return fig

def bokeh_vbar_chart(df,categories_col,value_col,title,color='#4F4478',width=600,height=380):
    rdf = df[[categories_col,value_col]]
    factors = list(rdf[categories_col])
    fig = bokeh.plotting.figure(title=title, width=width,height=height,x_range=bokeh.models.FactorRange(*factors))
    fig.vbar(bottom=0, top=rdf[value_col], x=factors , color=color, width=0.5, alpha=0.8)
    return fig


def bokeh_multi_hbar_chart(df,cat_col,value_list,width=400,height=300):
    chart_list=[]
    value_counter = len(value_list)
    if (value_counter <= 3):
        color_list=['#5154eb','#b2d235','#df9815']
    elif (value_counter <=10):
        color_list=bokeh.palettes.Category10[10]
    else:
        color_list=bokeh.palettes.Category20[20]
    for idx in range(0,value_counter):
        value_name = value_list[idx]
        pfig = bokeh_hbar_chart(df,cat_col,value_name,value_name,color=color_list[idx], width=width,height=height)
        chart_list.append(pfig)

    return chart_list



def bokeh_hist_chart(item_list,title,bins=100,width=400,height=300,legend_location='bottom_left'):
    fig =  bokeh.plotting.figure(width=width,height=height,title=title)
    lines_counter = len (item_list)
    if (lines_counter <=3):
        color_list=['#036564','red','navy']
    elif (lines_counter <=10):
        color_list=bokeh.palettes.Category10b[10]
    else:
        color_list=bokeh.palettes.Category20b[20]

    for idx in range(0,lines_counter):
        hist,edges = np.histogram(item_list[idx], density=True, bins=bins)
        fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],fill_color=color_list[idx], line_color="#033649",alpha=0.5)
    return fig

def show_grid(chart_list, num_in_row = 4):
    grid_render_idx = 0
    grid_render_matrix = []
    templist =[]
    for item in chart_list:
        templist.append(item)
        grid_render_idx +=1
        if (grid_render_idx == num_in_row):
            grid_render_matrix.append(templist)  #append in a new line
            templist =[]
            grid_render_idx =0
    if (len(templist) >0 ):
        grid_render_matrix.append(templist)

    bokeh.plotting.show(bokeh.layouts.gridplot(grid_render_matrix))

def show_column(chart_list):
    bokeh.plotting.show(bokeh.layouts.column(chart_list))

def show_row(chart_list):
    bokeh.plotting.show(bokeh.layouts.row(chart_list))




