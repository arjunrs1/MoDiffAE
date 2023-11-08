#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import matplotlib.lines as lines
import numpy as np
import os


#cls = [x for x in data["technique_cls"]]

#print(len([x for x in cls if x == 0]))
#print(len([x for x in cls if x == 1]))
#print(len([x for x in cls if x == 2]))
#print(len([x for x in cls if x == 3]))
#print(len([x for x in cls if x == 4]))

#for c in range(5):
#    print('###')
#    for g in ['9 kyu', '8 kyu', '7 kyu', '6 kyu', '5 kyu', '4 kyu', '3 kyu', '2 kyu', '1 kyu', '1 dan', '2 dan', '3 dan', '4 dan']:
#        print(len([x for x in data if x["technique_cls"] == c and x['grade'] == g]))
#exit()

#for i, c in enumerate(cls):
#    if c == 0:
#        mocap_visualization.from_array(positions[i])

#print(self._num_frames_in_video)

#print(min(self._num_frames_in_video))
#print(max(self._num_frames_in_video))
#mean = np.mean(self._num_frames_in_video)
#print(np.mean(self._num_frames_in_video))
#std = np.std(self._num_frames_in_video)
#print(np.std(self._num_frames_in_video))
#print(np.var(self._num_frames_in_video))
#print(len([n for n in self._num_frames_in_video if n > 75]))

datapath='/home/anthony/pCloudDrive/storage/data/master_thesis/karate_prep/'
npydatafilepath = os.path.join(datapath, "karate_motion_25_fps.npy")
data = np.load(npydatafilepath, allow_pickle=True)

pose = [x for x in data["joint_angles"]]
num_frames_in_video = [p.shape[0] for p in pose]
# TODO print averge and number of samples where there are more than 125 (5 seconds)
#print(self._num_frames_in_video)

#print(min(self._num_frames_in_video))
#print(max(self._num_frames_in_video))
#mean = np.mean(self._num_frames_in_video)
#print(np.mean(self._num_frames_in_video))
#std = np.std(self._num_frames_in_video)
#print(np.std(self._num_frames_in_video))
#print(np.var(self._num_frames_in_video))
#print(len([n for n in self._num_frames_in_video if n > 75]))

values = np.array(num_frames_in_video) / 25
print(values)
values = [v for v in values if v <= 3]
mean = np.mean(values)
std = np.std(values)



fig, ax = plt.subplots(figsize=(10, 5))
#fig, ax = plt.subplots()

#ax.set_xlim(0, 8)

# xlimit is at 7.436 (unprocessed) with 65 bins
# new number of bins = 65 * (new_xlimit / 7.436)

# 65 before processing, after calculate by formula above
num_bins = int(65 * (3.11 / 7.436)) #65 # replace 3.11 with correct new xlimit
print('hi', num_bins)
#plt.hist(values, bins=range(min(values), max(values) + 3/25, 3/25))
#bins = np.linspace(min(values), max(values), num_bins)#15)#35)
#print(bins)
#ax.set_ylim(0, 1000)
#ax.autoscale(False)
#n, bins, patches = ax.hist(values, bins=bins, color='tab:blue')#, density=True) #60))
#sns.histplot(data=values, kde=True, ax=ax, bins=num_bins, line_kws={'linestyle': 'dashed'})
sns.histplot(data=values, ax=ax, bins=num_bins)
min_ylim, max_ylim = plt.ylim()

#print('probability to fall between {0} and {1} :'.format(x1, x2), integrate.quad(distribution_function, x1, x2)[0])
#print(n, bins, patches)

#y = ((1 / (np.sqrt(2 * np.pi) * std)) *
#    np.exp(-0.5 * (1 / std * (bins - mean))**2))
#x = np.linspace(mean - 3*std, mean + 3*std, 100)
x = np.linspace(min(values), max(values), 100)
y = stats.norm.pdf(x, mean, std)
#print(x, y)
#print(mean, std)


ax2 = ax.twinx()

bin_width = (max(values) - min(values)) / num_bins
print(bin_width)
hist_area = len(values) * bin_width
print(hist_area)
print(ax.get_ylim())
new_lim = ax.get_ylim()[1] / hist_area
print(new_lim)
ax2.set_ylim(0, new_lim)


#ax2.autoscale(False)
l1, = ax2.plot(x, y ,'--', color='tab:orange', label='pdf')

# I checked that this actually completely aligns 
# with the probability density scale! i.e the
# graph has the same scaling as the kde on ax.
# So I can savely change the color of the kde funciton 
# to orange
sns.kdeplot(data=values, ax=ax2, linestyle='solid', color='tab:orange', label='test')
#sns.kdeplot(data=values, ax=ax2, linestyle='dotted', color='tab:orange', label='test')


red_patch = lines.Line2D(xdata=[], ydata=[], color='tab:orange', label='kde')

ax.legend(handles=[l1, red_patch])

#num_bins = 10

#ax.plot(x, y, '--', color='tab:orange')


ax.axvline(mean, color='k', linestyle='dashed')
ax.text(mean*1.05, max_ylim*0.9, '\u03BC')
ax.axvline(mean + std, color='k', linestyle='dashed')
ax.text((mean + std)*1.05, max_ylim*0.7, '\u03BC + \u03C3')
ax.axvline(mean - std, color='k', linestyle='dashed')
ax.text((mean - std)*0.525, max_ylim*0.7, '\u03BC - \u03C3')

ax.set_xlabel('Duration in seconds')
ax.set_ylabel('Number of participants', color='tab:blue')
ax2.set_ylabel('Probability density', color='tab:orange')

textstr = '\n'.join((
    r'$\mu=%.2f$' % (mean, ),
    r'$\sigma=%.2f$' % (std, )))
props = dict(boxstyle='round', facecolor='white', alpha=0.1)
ax.text(0.8525, 0.82, textstr,
    transform=ax.transAxes, verticalalignment='top', bbox=props)

# \u03BC
# \u03C3

#ax.autoscale(False)
#ax2.autoscale(False)

plt.autoscale(False)
plt.autoscale(False)

print(ax.get_ylim())
print(ax2.get_ylim())
print(ax.get_xlim())

print(mean, std)

plt.show()




#def main(desired_frequency, data_dir, output_dir, replace):
#    pass
    

#if __name__ == "__main__":
    



'''
# TODO print averge and number of samples where there are more than 125 (5 seconds)
        print(self._num_frames_in_video)

        print(min(self._num_frames_in_video))
        print(max(self._num_frames_in_video))
        mean = np.mean(self._num_frames_in_video)
        print(np.mean(self._num_frames_in_video))
        std = np.std(self._num_frames_in_video)
        print(np.std(self._num_frames_in_video))
        print(np.var(self._num_frames_in_video))
        print(len([n for n in self._num_frames_in_video if n > 75]))

        values = np.array(self._num_frames_in_video) / 25
        print(values)
        values = [v for v in values if v <= 3]
        mean = np.mean(values)
        std = np.std(values)

        

        fig, ax = plt.subplots(figsize=(10, 5))
        #fig, ax = plt.subplots()

        #ax.set_xlim(0, 8)
        
        # xlimit is at 7.436 (unprocessed) with 65 bins
        # new number of bins = 65 * (new_xlimit / 7.436)

        # 65 before processing, after calculate by formula above
        num_bins = int(65 * (3.11 / 7.436)) #65 # replace 3.11 with correct new xlimit
        print('hi', num_bins)
        #plt.hist(values, bins=range(min(values), max(values) + 3/25, 3/25))
        #bins = np.linspace(min(values), max(values), num_bins)#15)#35)
        #print(bins)
        #ax.set_ylim(0, 1000)
        #ax.autoscale(False)
        #n, bins, patches = ax.hist(values, bins=bins, color='tab:blue')#, density=True) #60))
        #sns.histplot(data=values, kde=True, ax=ax, bins=num_bins, line_kws={'linestyle': 'dashed'})
        sns.histplot(data=values, ax=ax, bins=num_bins)
        min_ylim, max_ylim = plt.ylim()

        #print('probability to fall between {0} and {1} :'.format(x1, x2), integrate.quad(distribution_function, x1, x2)[0])
        #print(n, bins, patches)

        #y = ((1 / (np.sqrt(2 * np.pi) * std)) *
        #    np.exp(-0.5 * (1 / std * (bins - mean))**2))
        #x = np.linspace(mean - 3*std, mean + 3*std, 100)
        x = np.linspace(min(values), max(values), 100)
        y = stats.norm.pdf(x, mean, std)
        #print(x, y)
        #print(mean, std)

        
        ax2 = ax.twinx()
        
        bin_width = (max(values) - min(values)) / num_bins
        print(bin_width)
        hist_area = len(values) * bin_width
        print(hist_area)
        print(ax.get_ylim())
        new_lim = ax.get_ylim()[1] / hist_area
        print(new_lim)
        ax2.set_ylim(0, new_lim)
        

        #ax2.autoscale(False)
        l1, = ax2.plot(x, y ,'--', color='tab:orange', label='pdf')

        # I checked that this actually completely aligns 
        # with the probability density scale! i.e the
        # graph has the same scaling as the kde on ax.
        # So I can savely change the color of the kde funciton 
        # to orange
        #sns.kdeplot(data=values, ax=ax2, linestyle='dashed', color='tab:orange', label='test')
        sns.kdeplot(data=values, ax=ax2, linestyle='solid', color='tab:orange', label='test')
        

        red_patch = lines.Line2D(xdata=[], ydata=[], color='tab:orange', label='kde')

        ax.legend(handles=[l1, red_patch])

        #num_bins = 10
        
        #ax.plot(x, y, '--', color='tab:orange')
        
        
        ax.axvline(mean, color='k', linestyle='dashed')
        ax.text(mean*1.05, max_ylim*0.9, '\u03BC')
        ax.axvline(mean + std, color='k', linestyle='dashed')
        #ax.axvline(mean + std, color='y', linestyle='dashed')
        ax.text((mean + std)*1.05, max_ylim*0.7, '\u03BC + \u03C3')
        #ax.axvline(mean - std, color='y', linestyle='dashed')
        ax.axvline(mean - std, color='k', linestyle='dashed')
        ax.text((mean - std)*0.525, max_ylim*0.7, '\u03BC - \u03C3')

        ax.set_xlabel('Duration in seconds')
        ax.set_ylabel('Number of participants', color='tab:blue')
        ax2.set_ylabel('Probability density', color='tab:orange')

        textstr = '\n'.join((
            r'$\mu=%.2f$' % (mean, ),
            r'$\sigma=%.2f$' % (std, )))
        props = dict(boxstyle='round', facecolor='white', alpha=0.1)
        ax.text(0.8525, 0.82, textstr,
            transform=ax.transAxes, verticalalignment='top', bbox=props)

        # \u03BC
        # \u03C3

        ax.autoscale(False)
        ax2.autoscale(False)

        #plt.autoscale(False)
        #plt.autoscale(False)

        print(ax.get_ylim())
        print(ax2.get_ylim())
        print(ax.get_xlim())

        print(mean, std)

        plt.show()

        exit()

'''