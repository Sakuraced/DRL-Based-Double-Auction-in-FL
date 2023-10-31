#Authors: Yanan Xu<ynxu15@gmail.com>

import sys
import importlib
importlib.reload(sys)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rc('font',family='Times New Roman',weight=100)

# ================================== data preparing ============================================================
x = np.arange(0., 5., 0.2)

y = []
y1 = x
y2 = x**2
y3 = x**3
y = [y1,y2,y3]
acc29=[76.00730895996094, 80.91976165771484, 85.01896667480469, 85.8065414428711, 86.47975158691406, 86.70832824707031, 85.60963439941406, 86.38796997070312, 88.46635437011719, 89.07424926757812, 88.32343292236328, 88.74459075927734, 88.92372131347656, 89.22064208984375, 89.03239440917969, 87.69623565673828, 89.0121841430664, 89.68315887451172, 89.23023223876953, 89.822509765625, 89.64720153808594, 89.79115295410156, 89.85210418701172, 90.187744140625, 89.80146026611328, 88.63410949707031, 90.48661804199219, 90.14511108398438, 90.0413589477539, 90.74486541748047, 90.70367431640625, 90.27558898925781, 90.03681945800781, 90.52886962890625, 90.93476104736328, 90.81583404541016, 91.27180480957031, 90.66580200195312, 90.66267395019531, 90.65240478515625, 90.46232604980469, 90.87027740478516, 89.83261108398438, 90.97330474853516, 89.85944366455078, 91.34605407714844, 90.6744155883789, 91.15811920166016, 90.7314682006836]
acc30=[76.00887298583984, 78.91304779052734, 81.71074676513672, 82.33946228027344, 83.91836547851562, 84.43096160888672, 83.94672393798828, 85.05487823486328, 84.85806274414062, 85.38325500488281, 85.14379119873047, 85.82379150390625, 85.31053161621094, 85.78184509277344, 85.49922180175781, 85.47834014892578, 86.75739288330078, 86.13814544677734, 86.78620147705078, 85.90650939941406, 86.97926330566406, 87.13127899169922, 86.64996337890625, 87.76509857177734, 88.08533477783203, 87.09907531738281, 87.17333221435547, 86.26811218261719, 87.43788146972656, 87.10643768310547, 86.77641296386719, 87.94580841064453, 87.80501556396484, 87.92010498046875, 87.44261932373047, 87.95050048828125, 87.75682830810547, 88.45613098144531, 87.5743179321289, 88.31939697265625, 88.43419647216797, 88.5368881225586, 88.61406707763672, 88.09642028808594, 88.33448028564453, 89.08732604980469, 87.76276397705078, 88.24681091308594, 89.0857925415039]
acc31=[79.16729736328125, 81.74347686767578, 81.82479858398438, 82.82573699951172, 83.59309387207031, 84.04986572265625, 84.28144836425781, 84.30872344970703, 84.09208679199219, 84.33417510986328, 84.64993286132812, 84.48939514160156, 84.69111633300781, 85.23442077636719, 85.29996490478516, 85.10774230957031, 84.55992126464844, 85.60795593261719, 84.61380004882812, 85.10961151123047, 84.81831359863281, 85.21015930175781, 85.52285766601562, 85.23302459716797, 84.64006042480469, 85.11161804199219, 85.15165710449219, 85.12327575683594, 84.7990951538086, 85.70188903808594, 85.58489227294922, 85.13170623779297, 84.69041442871094, 84.77323913574219, 85.56245422363281, 86.02024841308594, 85.06986999511719, 85.41047668457031, 85.05802917480469, 85.42984771728516, 85.23625183105469, 85.49015045166016, 85.210693359375, 85.40864562988281, 84.75502014160156, 85.48966979980469, 85.69481658935547, 85.40846252441406, 85.83853912353516]
acc32=[51.75739669799805, 50.90499496459961, 52.13897705078125, 65.85118865966797, 58.23900604248047, 72.96501159667969, 73.21200561523438, 76.66260528564453, 78.03526306152344, 77.07807159423828, 78.56465911865234, 76.31954956054688, 77.46147155761719, 78.25264739990234, 76.59425354003906, 77.52661895751953, 76.91746520996094, 79.77149200439453, 78.15556335449219, 78.96302795410156, 79.6838607788086, 79.36573791503906, 79.91390991210938, 80.95538330078125, 80.411865234375, 81.34001159667969, 80.43461608886719, 80.11858367919922, 80.90393829345703, 80.93079376220703, 82.47874450683594, 82.84976196289062, 82.48818969726562, 82.13468933105469, 83.04060363769531, 85.16416931152344, 82.74272918701172, 82.59638977050781, 83.986572265625, 83.5224380493164, 85.13685607910156, 84.94483184814453, 85.66474151611328, 84.70134735107422, 83.92902374267578, 85.7524642944336, 84.5958251953125, 86.10842895507812, 83.6543197631836]
round=[ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,48, 49, 50]
acc=[acc29,acc30,acc31,acc32]
ac=[70.81413536071777, 78.28444786071778, 83.17399635314942, 85.05468406677247, 84.8856029510498, 86.84562225341797, 87.73611793518066, 88.87132263183594, 89.05914649963378, 89.54415702819824, 88.94824676513672, 89.81794624328613, 90.14569969177246, 89.88314437866211, 89.64378623962402, 90.98297729492188, 91.1195297241211, 90.08912200927735, 90.87119750976562, 90.52462272644043, 90.15335540771484, 90.41052932739258, 90.71895484924316, 90.3033836364746, 90.91798210144043, 90.80820846557617, 90.5630657196045, 90.14287185668945, 90.84745635986329, 91.09644927978516, 90.48965568542481, 90.63772888183594, 90.87534561157227, 90.36753005981446, 90.96514778137207, 90.31478271484374, 90.77623023986817, 90.84489288330079, 89.90432090759278, 90.96159477233887, 90.21826438903808, 90.69261207580567, 90.50075454711914, 90.91342582702637, 90.28392219543457, 91.26566619873047, 90.00933685302735, 89.81239395141601, 90.37576141357422, 33.24121446609497, 43.3753381729126, 49.16898279190063, 55.07527084350586, 58.30265884399414, 62.612396240234375, 65.08110599517822, 66.10314846038818, 69.69140510559082, 70.19202270507813, 71.13763256072998, 72.04816303253173, 73.3455961227417, 74.19969272613525, 74.51800365447998, 76.3584020614624, 75.31021156311036, 75.41053771972656, 77.13603210449219, 76.9908037185669, 78.01341705322265, 77.14973316192626, 78.49146633148193, 79.42931671142578, 79.29727745056152, 79.52506732940674, 78.89956398010254, 79.50321998596192, 79.84259929656983, 80.10486602783203, 79.17895050048828, 80.36288223266601, 81.13963928222657, 81.06158065795898, 80.8451141357422, 81.58929901123047, 81.47674026489258, 81.01129837036133, 81.85636291503906, 82.09191780090332, 80.9705020904541, 82.3172706604004, 80.8742057800293, 81.98110389709473, 82.23659172058106, 82.5960132598877, 81.69333343505859, 82.2900779724121, 82.25928497314453]
simi=[0.7547839909791947, 0.7856211125850677, 0.8029748439788819, 0.8282754972577095, 0.8387909956276417, 0.8391856044530869, 0.8623758107423782, 0.8510716065764428, 0.8513404950499535, 0.868307477235794, 0.8756261840462685, 0.8758908629417419, 0.8698391482234001, 0.8869664534926415, 0.882393603771925, 0.8817783087491989, 0.8863407582044601, 0.8854909315705299, 0.8881886221468449, 0.8771314047276974, 0.8809413976967335, 0.8879366502165794, 0.8865529887378216, 0.8892537355422974, 0.9017179496586323, 0.9000930950045586, 0.8921477980911732, 0.8932537667453289, 0.8923573315143585, 0.8956426970660687, 0.8985889405012131, 0.8975692935287952, 0.9001459673047065, 0.8987848192453385, 0.9099127478897572, 0.9014029517769814, 0.8988295011222363, 0.8972342051565647, 0.9009563431143761, 0.9046037673950196, 0.8985090419650078, 0.9075377456843853, 0.9019553296267986, 0.9110799498856068, 0.900225818157196, 0.9083577327430248, 0.9067456088960171, 0.9025635376572609, 0.9056509383022785,0.38484812065174706, 0.4140883246534749, 0.4444750991306807, 0.4692305080200497, 0.49841340717516447, 0.5200588849030043, 0.5447565464597, 0.5580334577121233, 0.5816844177873511, 0.5901803622120305, 0.6107156496298941, 0.6136947658501173, 0.6338699113381536, 0.6433537122450377, 0.6446792204129068, 0.6557812888371317, 0.6667249409775985, 0.6700872744384565, 0.6778458683114302, 0.6849721522707688, 0.6920447101718501, 0.69618573596603, 0.7079291930324153, 0.7047586023807526, 0.7098674984354721, 0.7201869362278989, 0.7206398505913584, 0.7286627562422502, 0.727043721864098, 0.7355659459766589, 0.742438234780964, 0.7440960118645116, 0.745214246135009, 0.7466227208313189, 0.7531399529231222, 0.7522462430753206, 0.7595019701280092, 0.7620692444475073, 0.7641217178420017, 0.7628509615596972, 0.7676459779864864, 0.7766203033296686, 0.7745156404219176, 0.7849747206035413, 0.7801915134254255, 0.7804252728035576, 0.7872347844274421, 0.7863475655254565, 0.7903043884980051]

# ================================== settings =================================================================
figureName = 'lineChart.pdf'

# -------------------------------------- style -----------------------------------------------------------------
# print(plt.style.available)     # print all options
plt.style.use('seaborn-paper')        # set your style

# style option
# ['bmh', 'seaborn-poster', 'seaborn-talk', 'seaborn-dark-palette', 'seaborn-muted', 'seaborn-colorblind',
# 'classic', 'seaborn-white', 'seaborn-pastel', 'dark_background', 'seaborn-ticks', 'seaborn-whitegrid',
# 'seaborn-darkgrid', 'fivethirtyeight', 'seaborn-notebook', 'seaborn-dark', 'grayscale', 'ggplot',
# 'seaborn-deep', 'seaborn-paper', 'seaborn-bright']

#----------------------------------- font size ----------------------------------------------------------------
FONT_SIZE_LEGEND = 20
FONT_SIZE_AXIS = 15
FONT_SIZE_LABEL = 26
FONT_SIZE_TITLE = 26

#----------------------------------- axies setting ----------------------------------------------------------------
NUM_LINES = 3

X_LIM = [2, 50]
Y_LIM = [75,92.5]

LINE_WIDTH_AXIS = 1.5

TITLE = 'Dataset with distribution 1'
LABEL_X = r'Round'
LABEL_Y = r'Accurancy'
LABEL_TITLE = 'Dataset with distribution 1'
FLAG_GRID = True

#----------------------------------- line setting ----------------------------------------------------------------
LINE_WIDTH = 2
MAKRER_SIZE = 7

# default color is black
LINE_COLOR = []
for i in range(NUM_LINES):
    LINE_COLOR.append(np.array([0,0,0])/255.0)

# LINE_COLOR[0] = np.array([86, 24, 27])/255
# LINE_COLOR[1] = np.array([214, 86, 42])/255
# LINE_COLOR[2] = np.array([60, 60, 60])/255
# LINE_COLOR[3] = np.array([90, 90, 90])/255

LINE_COLOR[0] = np.array([102, 0, 0])/255
LINE_COLOR[1] = np.array([1, 46, 137])/255
LINE_COLOR[2] = np.array([19, 103, 107])/255
#LINE_COLOR[3] = np.array([202, 98, 4])/255
# LINE_COLOR[4] = np.array([210, 210, 0])/255
# LINE_COLOR[5] = np.array([103, 12, 203])/255
# LINE_COLOR[6] = np.array([213, 71, 115])/255

# default style is '-'
# '-', '--', '-.', ':', 'steps'
LINE_STYLE = []
for i in range(NUM_LINES):
    LINE_STYLE.append('-')
LINE_STYLE = [':','--','-.','dashed']

# default marker shape is '^'
MARKER_FACE_FLAG = False
MARKER_SHAPE = []
for i in range(NUM_LINES):
    MARKER_SHAPE.append('p')

MARKER_SHAPE[1] = 'o';
MARKER_SHAPE[2] = 'h';
#MARKER_SHAPE[3] = '+';
# MARKER_SHAPE[4] = 'x';
# MARKER_SHAPE[5] = '*';
# MARKER_SHAPE[6] = '^';

# . Point marker
# , Pixel marker
# o Circle marker
# v, ^, <, > Triangle markers
# 1, 2, 3, 4, Tripod marker
# s Square marker
# p Pentagon marker
# * Star marker
# h, H Hexagon marker
# D, d Diamond marker
# | Vertical line
# _ Horizontal line
# + Plus marker
# x Cross marker

# by default, all marker faces have a color of 'w'
MARKER_FACE_COLOR = []
for i in range(NUM_LINES):
    MARKER_FACE_COLOR.append(LINE_COLOR[i])


#----------------------------------- legend setting ----------------------------------------------------------------
# legend position
# 'best' : 0, (only implemented for axes legends)
# 'upper right' : 1,
# 'upper left' : 2,
# 'lower left' : 3,
# 'lower right' : 4,
# 'right' : 5,
# 'center left' : 6,
# 'center right' : 7,
# 'lower center' : 8,
# 'upper center' : 9,
# 'center' : 10,
LEGEND_POSITION = 2

LEGEND_BOX_FLAG = False

#by default, all legend texts are set to 'linei'
LEGEND_TEXT = []
for i in range(NUM_LINES):
    LEGEND_TEXT.append('Line'+str(i))
LEGEND_TEXT[0] = 'RL'
LEGEND_TEXT[1] = 'Distributed group'
LEGEND_TEXT[2] = '0 Recommmendation'

# ================================== plot figure ============================================================
fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
#plt.axes([0.15, 0.15, 0.75, 0.75])

lines = []
'''
for i in range(NUM_LINES):
    line, = plt.plot(round, acc[i], linestyle = LINE_STYLE[i], linewidth = LINE_WIDTH, color = LINE_COLOR[i],
                    marker = MARKER_SHAPE[i], markeredgecolor = MARKER_FACE_COLOR[i].tolist(),
                    markeredgewidth=1, markerfacecolor=MARKER_FACE_COLOR[i].tolist(),
                    markersize= MAKRER_SIZE) # alpha is the transparency
'''
plt.scatter(ac,simi,marker = MARKER_SHAPE[0],c= MARKER_FACE_COLOR[1].tolist())
if LABEL_TITLE != None:
    plt.title(LABEL_TITLE, fontsize=FONT_SIZE_TITLE, fontname='Times New Roman' )

if LABEL_X != None:
    plt.xlabel(LABEL_X, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')
if LABEL_Y != None:
    plt.ylabel(LABEL_Y, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')

# grid
plt.grid(FLAG_GRID)
#ax = plt.gca()
ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
plt.tick_params(which='minor', length=2, color='k')

# for the minor ticks, use no labels; default NullFormatter
#ax.xaxis.set_minor_locator(minorLocator)

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(FONT_SIZE_AXIS)
    tick.label1.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(FONT_SIZE_AXIS)
    tick.label1.set_fontname('Times New Roman')

for label in ax.xaxis.get_ticklabels():
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')
for label in ax.yaxis.get_ticklabels():
    #label.set_color('red')
    #label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')

ax.spines['bottom'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['left'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['top'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['right'].set_linewidth(LINE_WIDTH_AXIS)
for line in ax.xaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')
    line.set_markersize(4)  # line length
    line.set_markeredgewidth(1.2) # line width
for line in ax.yaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')     # line color
    line.set_markersize(4)      # line length
    line.set_markeredgewidth(1.2) # line width
plt.show()
# plt.yscale('linear') # linear, log, logit, or symlog
# plt.xscale('linear') # linear, log, logit, or symlog

# Adjust the subplot layout, because the logit one may take more space
# than usual
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
# wspace=0.35)

# add one text to the plot
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

# add one annotate to the plot
# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )

# legend
'''matplotlib.legend.Legend(parent, handles, labels, loc=None, numpoints=None, markerscale=None,
markerfirst=True, scatterpoints=None,
scatteryoffsets=None, prop=None, fontsize=None, borderpad=None,
labelspacing=None, handlelength=None,
handleheight=None, handletextpad=None, borderaxespad=None,
columnspacing=None, ncol=1, mode=None, fancybox=None,
shadow=None, title=None, framealpha=None,
edgecolor=None, facecolor=None, bbox_to_anchor=None,
bbox_transform=None, frameon=None, handler_map=None)'''
#plt.legend(lines,LEGEND_TEXT, loc =LEGEND_POSITION, numpoints=1, markerscale=1, fontsize=FONT_SIZE_LEGEND, ncol =2, bbox_to_anchor=[0.01,0.99] )

lg = plt.legend(labels = LEGEND_TEXT, loc =LEGEND_POSITION, numpoints=1, markerscale=1, fontsize=FONT_SIZE_LEGEND, ncol =1, bbox_to_anchor=[0.01,0.99] )
if not LEGEND_BOX_FLAG:
    lg.draw_frame(False)

# matplotlib.text.Text instances
# for t in leg.get_texts():
#     t.set_fontsize(FONT_SIZE_LEGEND)  # the legend text fontsize
#     print(t.get_fontproperties())
#     fp = t.get_fontproperties()
#     fp = mpl.font_manager.FontProperties(family='times new roman', style='normal', weight=100, size=FONT_SIZE_LEGEND)
#     t.set_font_properties(fp)
#
# # set line width of legend. matplotlib.lines.Line2D instances
# for l in leg.get_lines():
#     l.set_linewidth(1.5)  # the legend line width

# Title and label
if LABEL_TITLE != None:
    plt.title(LABEL_TITLE, fontsize=FONT_SIZE_TITLE, fontname='Times New Roman' )

if LABEL_X != None:
    plt.xlabel(LABEL_X, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')
if LABEL_Y != None:
    plt.ylabel(LABEL_Y, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')

# grid
plt.grid(FLAG_GRID)

# axises
if X_LIM != None:
    plt.xlim(X_LIM)
if Y_LIM != None:
    plt.ylim(Y_LIM)
# major ticks
#plt.xticks([0,1,2,3,4,5])
# plt.yticks([0.0, 0.2,0.4,0.6],
# 			['-5',  '-3',  '-1',  '1'])

# minor ticks
#ax = plt.gca()
ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
plt.tick_params(which='minor', length=2, color='k')

# for the minor ticks, use no labels; default NullFormatter
#ax.xaxis.set_minor_locator(minorLocator)

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(FONT_SIZE_AXIS)
    tick.label1.set_fontname('Times New Roman')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(FONT_SIZE_AXIS)
    tick.label1.set_fontname('Times New Roman')

for label in ax.xaxis.get_ticklabels():
    label.set_color('red')
    label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')
for label in ax.yaxis.get_ticklabels():
    #label.set_color('red')
    #label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')

ax.spines['bottom'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['left'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['top'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['right'].set_linewidth(LINE_WIDTH_AXIS)
for line in ax.xaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')
    line.set_markersize(4)  # line length
    line.set_markeredgewidth(1.2) # line width
for line in ax.yaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')     # line color
    line.set_markersize(4)      # line length
    line.set_markeredgewidth(1.2) # line width

#tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

foo_fig = plt.gcf()
foo_fig.savefig(figureName)
plt.show()
