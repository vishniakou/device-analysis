import glob
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import constants

mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.size'] = 14
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['lines.linewidth'] = 2
#    mpl.rcParams['axes.formatter.use_mathtext'] = True
#    mpl.rcParams['text.usetex'] = True
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.transparent'] = True


def fit_id_vd(data):
    mu = 0.1
    vt = 2

    popt, pcov = curve_fit(id_vd, data, data['Id'], p0=[mu, vt])
    return popt, pcov


def id_vd(data, mu, vt):
    # data must be in format [vd, vg, id]
    vd = data['Vd']
    vg = data['Vg']


    # C2 device parameters
    tox = 900E-9  # dielectric thickness in meters
    Ko = 50 # dielectric constant
    WoverL = 10  # width to length ratio


    Cox = Ko*constants.epsilon_0/tox

    Id = WoverL*Cox*mu*(np.multiply(vg - vt, vd) - 0.5*np.square(vd))
    # Id should be zero where Vg < Vt
    Id[vg < vt] = 0
    # Take care of saturation
    vdsat = vd >= (vg - vt)
    Idsat = 0.5*WoverL*mu*Cox*np.square(vg - vt)
    Id[vdsat & (vg > vt)] = Idsat
#    Id[vdsat] = 0.5*WoverL*mu*Cox*np.square(vg[vdsat]-vt)

    return Id


def plot_id_vd(data, linespec):
    """Use existing figure to plot Id vs Vd."""
    vg_values = data['Vg'].unique()
    # set engineering notation for y axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for vg in vg_values:
        # select only data at this Vg
        data_subset = data.ix[data['Vg'] == vg]
        x = data_subset['Vd']
        y = data_subset['Id']

        plt.plot(x, y, linespec, label=('Vg=' + str(vg) + 'V'))


def plot_id_vg(data, linespec):
    """Use existing figure to plot Id vs Vd."""
    vd_values = data['Vd'].unique()
    # set engineering notation for y axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for vd in vd_values:
        # select only data at this Vg
        data_subset = data.ix[data['Vd'] == vd]
        x = data_subset['Vg']
        y = data_subset['Id']

        plt.plot(x, y, linespec, label=('Vd=' + str(vd) + 'V'))
#    xlim = plt.gca().get_xlim()
#    ylim = plt.gca().get_ylim()
#    plt.gca().set_xlim([0, xlim[1]])
#    plt.gca().set_ylim([0, ylim[1]])


def plot_id_vd_compare(data, data_fit):
    """Use existing figure to plot Id vs Vd."""
    vg_values = data['Vg'].unique()
    # set engineering notation for y axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for vg in vg_values:
        # select only data at this Vg
        data_subset = data.ix[data['Vg'] == vg]
        data_fit_subset = data_fit[data_fit['Vg'] == vg]

        plot = plt.plot(data_subset['Vd'], data_subset['Id'], '-',
                        label=('Vg=' + str(vg) + 'V'))
        plt.plot(data_fit_subset['Vd'], data_fit_subset['Id'], '--',
                 color=plot[0].get_color())
    plt.xlabel('Vd, V')
    plt.ylabel('Current, A')


def plot_id_vg_compare(data, data_fit):
    """Use existing figure to plot Id vs Vd."""
    vd_values = data['Vd'].unique()
    # set engineering notation for y axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for vd in vd_values:
        # select only data at this Vg
        data_subset = data.ix[data['Vd'] == vd]
        data_fit_subset = data_fit[data_fit['Vd'] == vd]

        plot = plt.plot(data_subset['Vg'], data_subset['Id'], '-',
                        label=('Vd=' + str(vd) + 'V'))
        plt.plot(data_fit_subset['Vg'], data_fit_subset['Id'], '--',
                 color=plot[0].get_color())
    plt.xlabel('Vg, V')
    plt.ylabel('Current, A')


def extract_from_file(filename):
    data = pd.io.parsers.read_csv(filename)
    data = data.reset_index().drop('index', axis=1)

    diff = data.ix[1, ['Vd', 'Vg']] - data.ix[0, ['Vd', 'Vg']]
    changing = data.columns[np.nonzero(diff)[0][0]]

    data = data[data['Vd'] >= 0]

    popt, pcov = fit_id_vd(data[data['Vg'] > 0])

    print('Mobility = %.2f cm2/Vs\tVt = %.2f V' % (popt[0]*1E4, popt[1]))

#    plot_id_vd(data, '-')
#    plt.legend(loc="upper left")
    fitted = id_vd(data, *popt)
    fitted.name = 'Id'
    data_fit = pd.concat([data[['Vg', 'Vd']], fitted.to_frame()], axis=1)

    if changing == 'Vd':
        # proceed analyzing Id-Vd measurement
        print('Id-Vd')
        plot_id_vd_compare(data, data_fit)
    elif changing == 'Vg':
        # proceed analyzing Id-Vg measurement
        print('Id-Vg')
        plot_id_vg_compare(data, data_fit)


def process_folder(folder):
    files = glob.glob(folder + '/*.msgpack')

#    tft = pd.DataFrame(columns=['device', 'pad', 'ion', 'ioff', 'mu', 'vt'])
    tft_info = pd.DataFrame()

    for f in files:
        name = os.path.split(f)[-1]  # extract file name from full file path
        match = re.search(r'Dev(\d)_Pad(\d)_run(\d)', name)
        if not match:  # filename does not follow the pattern above, skip file
            continue
        (dev, pad, run) = match.groups()

#        tft_info = {'device': dev, 'pad': pad, 'ion': 0,
#                    'ioff': 0}
#        print(name, '\t', dev, ' ', ' ', pad, ' ', run, '|')
        data = pd.read_msgpack(f)
        data = data.reset_index().drop('index', axis=1)

        data_temp = data.drop('Id', axis=1)
        diff = data_temp.ix[1, ['Vd', 'Vg']] - data_temp.ix[0, ['Vd', 'Vg']]
#        print(diff)
#        print(data.columns)
#
#        print(np.nonzero(diff)[0][0])
        changing = data_temp.columns[np.nonzero(diff)[0][0]]
#        print(changing)
#        raise "Ex"
        if changing == 'Vd':
            # proceed analyzing Id-Vd measurement
#            print('Id-Vd')
            popt, pcov = fit_id_vd(data[data['Vg'] > 0])
            mobility = popt[0]*1E4
            vt = popt[1]

            ion = data['Id'].max()

#            print('Mobility = %.2f cm2/Vs\tVt = %.2f V' % (popt[0]*1E4, popt[1]))
            print('%s\t%s\t%.2f\t%.2f\t%.2e' % (dev, pad, mobility, vt, ion))
            tft_info = tft_info.append({'Device': dev, 'Pad': pad,
                                        'Mobility': mobility, 'Vt': vt,
                                        'Ion': ion}, ignore_index=True)
#    plot_id_vd(data, '-')
#    plt.legend(loc="upper left")
            fitted = id_vd(data, *popt)
            fitted.name = 'Id'
            data_fit = pd.concat([data[['Vg', 'Vd']], fitted.to_frame()],
                                  axis=1)
            plt.figure()
            plot_id_vd_compare(data, data_fit)
            plt.title('Dev %s Pad %s' % (dev, pad))
    # Create various plots
    figure_vt = plt.figure()
    plt.title('Vt')
    plt.ylabel('Voltage, V')
    plt.xlabel('Channel')
#    plt.legend()
    figure_mobility = plt.figure()
    plt.title('Mobility')
    plt.xlabel('Channel')
#    plt.legend()
    figure_ion = plt.figure()
    plt.title('On current')
    plt.ylabel('Current, A')
    plt.xlabel('Channel')
#    plt.legend()
    for device in tft_info['Device'].unique():
        data_select = tft_info[tft_info['Device'] == device]
        label = 'Device' + device

        plt.figure(figure_vt.number)
        plt.plot(data_select['Pad'], data_select['Vt'], label=label)

        plt.figure(figure_mobility.number)
        plt.plot(data_select['Pad'], data_select['Mobility'], label=label)

        plt.figure(figure_ion.number)
        plt.plot(data_select['Pad'], data_select['Ion'], label=label)

#        elif changing == 'Vg':
#            # proceed analyzing Id-Vg measurement
#            print('Id-Vg')
#            plot_id_vg_compare(data, data_fit)
    plt.figure(figure_vt.number)
    plt.legend(loc='lower left')
    plt.figure(figure_mobility.number)
    plt.legend(loc='upper left')
    plt.figure(figure_ion.number)
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    process_folder()
