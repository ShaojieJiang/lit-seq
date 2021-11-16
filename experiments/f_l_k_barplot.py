# import numpy as np
# import matplotlib.pyplot as plt
# data = [[30, 25, 50, 20],
# [40, 23, 51, 17],
# [35, 22, 45, 19]]
# X = np.arange(4)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

import matplotlib.pyplot as plt
import numpy as np


labels = ['DD', 'PC', 'ED', 'WoW', 'BST']

all_p = [0.59, 0.62, 0.74, 0.59, 0.21]
all_s = [0.56, 0.56, 0.71, 0.55, 0.18]

fl3_p = [0.65, 0.77, 0.74, 0.65, 0.26]
fl3_s = [0.66, 0.76, 0.71, 0.66, 0.24]

fl2_p = [0.70, 0.85, 0.76, 0.74, 0.28]
fl2_s = [0.72, 0.79, 0.73, 0.73, 0.27]

fl1_p = [0.81, 0.92, 0.89, 0.86, 0.35]
fl1_s = [0.80, 0.88, 0.86, 0.83, 0.34]

men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(2 * len(labels), step=2)  # the label locations
width = 0.17  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*4, all_p, width, label='All-P')
rects2 = ax.bar(x - width*3, fl3_p, width, label='F&L 3-P')
rects3 = ax.bar(x - width*2, fl2_p, width, label='F&L 2-P')
rects4 = ax.bar(x - width*1, fl1_p, width, label='F&L 1-P')
rects5 = ax.bar(x + width*1, all_s, width, label='All-S')
rects6 = ax.bar(x + width*2, fl3_s, width, label='F&L 3-S')
rects7 = ax.bar(x + width*3, fl2_s, width, label='F&L 2-S')
rects8 = ax.bar(x + width*4, fl1_s, width, label='F&L 1-S')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson/Spearman Correlations')
# ax.set_title('RDEP correlations with RD labels')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig("flk.pdf")
plt.show()
print('done')