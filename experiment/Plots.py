import matplotlib.pyplot as plt
import seaborn as sns

def init_plotting():
    plt.rcParams["figure.dpi"] = 300               # 输出图片dpi
    plt.rcParams["figure.subplot.left"]=0.125      # 子图左间距
    plt.rcParams["figure.subplot.right"]=0.9       # 子图右间距                             
    plt.rcParams["figure.subplot.bottom"]=0.1      # 子图下边距          
    plt.rcParams["figure.subplot.top"]=0.9         # 子图上边距       
    plt.rcParams["figure.subplot.wspace"]=0.25     # 子图之间的水平距离          
    plt.rcParams["figure.subplot.hspace"]=0.65      # 子图之间的垂直距离          
    plt.rcParams["font.family"]="Times New Roman"  # 使用字体
    plt.rcParams["font.style"]="normal"            # 字体格式 normal italic oblique
    plt.rcParams["font.weight"]=400                # 字体是否加粗 normal(400) bold(700)  100 200 300 ...900  
    plt.rcParams["font.size"]=12                   # 字体大小
    plt.rcParams["lines.linewidth"]=2              # 线宽
    plt.rcParams["lines.markersize"]=8             # marker 大小               
    plt.rcParams["axes.linewidth"]=2               # 子图边框大小
    plt.rcParams["axes.titlesize"]=24              # 子图标题大小
    plt.rcParams["axes.titleweight"]=600           # 子图标题是否加粗 normal(400) bold(700)  100 200 300 ...900
    plt.rcParams["axes.titlepad"]=8                # 子标题与子图之间的padding
    plt.rcParams["axes.labelsize"]=24              # 坐标轴标签大小
    plt.rcParams["axes.labelpad"]=2                # 坐标轴标签与坐标轴之间的pad
    plt.rcParams["axes.labelweight"]=600           # 坐标轴标签是否加粗 normal(400) bold(700)  100 200 300 ...900
    plt.rcParams["axes.labelcolor"]="k"            # 坐标轴标签颜色
    plt.rcParams["axes.spines.left"]="True"        # 是否显示左轴
    plt.rcParams["axes.spines.bottom"]="True"      # 是否显示右轴
    plt.rcParams["axes.spines.top"]="True"         # 是否显示上轴
    plt.rcParams["axes.spines.right"]="True"       # 是否显示下轴
    plt.rcParams["xtick.major.size"]=7.5           # x轴主刻度大小
    plt.rcParams["xtick.minor.size"]=5.5           # x轴次刻度大小
    plt.rcParams["xtick.major.width"]=2            # x轴主刻度宽度
    plt.rcParams["xtick.minor.width"]=1            # x轴次刻度宽度
    plt.rcParams["xtick.major.pad"]=4              # x轴主刻度标签距离
    plt.rcParams["xtick.minor.pad"]=3              # x轴次刻度标签距离
    plt.rcParams["xtick.color"]="k"                # x轴刻度颜色
    plt.rcParams["xtick.labelsize"]=16             # x轴刻度标签大小        
    plt.rcParams["xtick.direction"]="out"          # x轴刻度方向 in out inout
    plt.rcParams["xtick.minor.visible"]="False"     # x轴是否显示次刻度 
    plt.rcParams["ytick.major.size"]=7.5           # y轴主刻度大小
    plt.rcParams["ytick.minor.size"]=5.5           # y轴次刻度大小
    plt.rcParams["ytick.major.width"]=2            # y轴主刻度宽度
    plt.rcParams["ytick.minor.width"]=1            # y轴次刻度宽度
    plt.rcParams["ytick.major.pad"]=4              # y轴主刻度标签距离
    plt.rcParams["ytick.minor.pad"]=3              # y轴次刻度标签距离
    plt.rcParams["ytick.color"]="k"                # y轴刻度颜色
    plt.rcParams["ytick.labelsize"]=16             # y轴刻度标签大小        
    plt.rcParams["ytick.direction"]="out"          # y轴刻度方向 in out inout
    plt.rcParams["ytick.minor.visible"]="False"     # y轴是否显示次刻度 
    plt.rcParams["legend.loc"]="best"              # Legend摆放位置  "best" "upper right" "upper left" "lower left" "lower right" "right" "center left" "center right" "lower center" "upper center" "center"
    plt.rcParams["legend.frameon"]="True"          # 是否显示legend边框
    plt.rcParams["legend.framealpha"]=0.8          # Legend 透明度
    plt.rcParams["legend.fontsize"]=16             # legend字体大小
    plt.rcParams["legend.columnspacing"]=2.0       # legend列间距

    plt.rcParams['mathtext.default'] = 'regular'

    colors = ["#ff595e","#ff924c","#ffca3a","#c5ca30","#8ac926","#52a675","#1982c4","#4267ac","#6a4c93","#d677b8", "#222222"]
    my_cmap = sns.color_palette(colors)
    sns.set_palette(my_cmap)