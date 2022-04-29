#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import os
from collections import defaultdict
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from obspy.geodetics import gps2dist_azimuth
from obspy.imaging.beachball import beach

from . import logger
from .util import get_cmt_par, get_trwin_tag
from .measure import _envelope

# earth half circle
EARTH_HC, _, _ = gps2dist_azimuth(0, 0, 0, 180)


def _plot_new_seismogram_sub(trwin, outputdir, cmtsource, figure_format):
    obsd = trwin.datalist['obsd']
    synt = trwin.datalist['synt']
    new_synt = trwin.datalist['new_synt']

    station = obsd.stats.station
    network = obsd.stats.network
    channel = obsd.stats.channel
    location = obsd.stats.location
    outputfig = os.path.join(outputdir, "%s.%s.%s.%s.%s" % (
        network, station, location, channel, figure_format))

    if cmtsource is None:
        offset = 0
    else:
        offset = obsd.stats.starttime - cmtsource.cmt_time
    times = [offset + obsd.stats.delta*i for i in range(obsd.stats.npts)]

    fig = plt.figure(figsize=(15, 5))

    plt.rcParams.update({'font.size': 13,
                         'lines.linewidth': 1.5})

    # plot seismogram
    ax1 = plt.subplot(211)
    ax1.plot(times, obsd.data, color="black", linewidth=0.8, alpha=0.6,
             label="obsd")
    ax1.plot(times, synt.data, color="blue", linewidth=1,
             label="synt")
    ax1.plot(times, new_synt.data, color="red", linewidth=1,
             label="new synt")
    ax1.set_xlim(times[0], times[-1])
    ax1.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})

    # Setting top left corner text manually
    fontsize = 11
    ax1.text(0.005, 0.8,
             "Network: %2s    Station: %s\n"
             "Location: %2s  Channel: %3s" %
             (network, station, location, channel),
             fontsize=fontsize,
             transform=ax1.transAxes)

    for win in trwin.windows:
        left = win[0] + offset
        right = win[1] + offset
        re = Rectangle((left, plt.ylim()[0]), right - left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue",
                       alpha=0.25)
        plt.gca().add_patch(re)

    # plot envelope
    ax2 = plt.subplot(212)
    ax2.plot(times, _envelope(obsd.data), color="black", linewidth=0.8,
             alpha=0.6, label="obsd")
    ax2.plot(times, _envelope(synt.data), color="blue", linewidth=1,
             label="synt")
    ax2.plot(times, _envelope(new_synt.data), color="red", linewidth=1,
             label="new synt")
    ax2.set_xlim(times[0], times[-1])

    ax2.set_xlabel("Time [s]", fontsize=13)

    for win in trwin.windows:
        left = win[0] + offset
        right = win[1] + offset
        re = Rectangle((left, plt.ylim()[0]), right - left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue",
                       alpha=0.25)
        plt.gca().add_patch(re)

    logger.info("output figname: %s" % outputfig)
    ax2.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})
    plt.savefig(outputfig)
    plt.close(fig)


def plot_seismograms(data_container, outputdir, cmtsource=None,
                     figure_format="png"):
    """
    Plot the new synthetic and old synthetic data together with data.
    So we can see the how the seimogram changes after inversion.
    """
    # make a check
    if 'new_synt' not in data_container.trwins[0].datalist.keys():
        return "New synt not generated...Can't plot"

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    logger.info("Plotting observed, synthetics and windows to dir: %s"
                % outputdir)
    for trwin in data_container:
        _plot_new_seismogram_sub(trwin, outputdir, cmtsource,
                                 figure_format)


class PlotStats(object):
    """ plot histogram utils"""

    def __init__(self, data_container, metas, outputfn, var_reduction):
        self.data_container = data_container
        self.metas = metas
        self.outputfn = outputfn
        self.var_reduction = var_reduction
        self.metas_sort = None
        self.key_map = None

    def sort_metas(self):
        metas_sort = defaultdict(list)
        key_map = defaultdict(set)
        """ sort metas into different categories for future plotting """
        for trwin, meta in zip(self.data_container, self.metas):
            comp = trwin.channel
            cat_name = get_trwin_tag(trwin)
            key_map[comp].add(cat_name)
            metas_sort[cat_name].append(meta)

        self.metas_sort = metas_sort
        self.key_map = key_map

    @staticmethod
    def plot_stats_histogram_one_entry(pos, cat, vtype, data_b, data_a,
                                       num_bin, var_reduction):
        plt.subplot(pos)
        plt.xlabel(vtype, fontsize=15, fontweight='bold')
        plt.ylabel(cat, fontsize=15)
        if vtype == "CC":
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
        elif vtype == r"$\chi$":
            ax_min = 0.0
            ax_max = max(max(data_b), max(data_a))
        else:
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
            abs_max = max(abs(ax_min), abs(ax_max))
            ax_min = -abs_max
            ax_max = abs_max

        MU1 = np.mean(data_b)
        MU2 = np.mean(data_a)
        STD1 = np.std(data_b)
        STD2 = np.std(data_a)

        binwidth = (ax_max - ax_min) / num_bin
        plt.hist(
            data_b, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
            facecolor='blue', alpha=0.3)
        plt.hist(
            data_a, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
            facecolor='red', alpha=0.5)
        if vtype == r"$\chi$":
            plt.yscale('log')
            #MU1 *= len(data_a)
            #MU2 *= len(data_a)
        if vtype in ["CC","TS","dlnA"]:
            annotation_string = r'$\mu: %.2f\to%.2f$' %(MU1,MU2) + '\n' + r'$\sigma: %.2f\to%.2f$' %(STD1,STD2) 
            plt.annotate(annotation_string, xy=(0.05, 0.88), xycoords='axes fraction', fontsize =12)
        elif vtype in [r"$\chi$"]:
            percentage = '{:.2f}%'.format(100 * (var_reduction[0]))
            percentage1 = '{:.2f}%'.format(100 * (var_reduction[1]))
            #plt.annotate('$\mu={:.2f}, {:.2f}$\n$\sigma={:.2f}, {:.2f}$\n$\Delta Var={:.2f}% $'.format(MU1,MU2,STD1,STD2,var_reduction), xy=(0.5, 0.81), xycoords='axes fraction', fontsize =12)
            annotation_string = r'$\mu: %.2f\to%.2f$' %(MU1,MU2) + '\n' + r'$\sigma: %.2f\to%.2f$' %(STD1,STD2) 
            plt.annotate(annotation_string, xy=(0.5, 0.81), xycoords='axes fraction', fontsize =12)
            plt.annotate('$\Delta (\sum \chi) = {} $%\n$\Delta \phi = {}$%'.format(percentage,percentage1), xy=(0.5, 0.60), xycoords='axes fraction', fontsize =12)
        elif vtype not in ['CC']:
            annotation_string = r'$\mu: %.2f\to%.2f$' %(MU1,MU2) + '\n' + r'$\sigma: %.2f\to%.2f$' %(STD1,STD2) 
            plt.annotate(annotation_string, xy=(0.5, 0.88), xycoords='axes fraction', fontsize =12)

    def extract_metadata(self, cat_name, meta_varname):
        data_old = []
        data_new = []
        cat_data = self.metas_sort[cat_name]
        for meta in cat_data:
            data_old.extend(meta.prov["synt"][meta_varname])
            data_new.extend(meta.prov["new_synt"][meta_varname])

        return data_old, data_new

    def plot_stats_histogram_one_category(
            self, G, irow, cat_name, vtype_list, num_bins, vtype_dict, out_metric, var_reduction):

        for var_idx, varname in enumerate(vtype_list):
            meta_varname = vtype_dict[varname]
            data_before, data_after = \
                self.extract_metadata(cat_name, meta_varname)
            tmp = []
            tmp.append(varname)
            tmp.append(data_before)
            tmp.append(data_after)
            out_metric.append(tmp)
            self.plot_stats_histogram_one_entry(
                G[irow, var_idx], cat_name, varname, data_before, data_after,
                num_bins[var_idx], var_reduction)

    def plot_stats_histogram(self):
        """
        Plot histogram of tshift, cc, power, cc_amplitude_ratio,
        waveform misfit values before and after inversion inside
        windows.

        :return:
        """
        import json
        #vtype_list = ['time shift', 'cc',
        #              'power_l1_ratio(dB)', 'power_l2_ratio(dB)',
        #              'CC amplitude ratio(dB)', 'chi']
        #vtype_list = ['TS', 'CC',
        #              'power_l1_ratio(dB)', 'dlnA',
        #              'CC amplitude ratio(dB)', r"$\chi$"]
        vtype_list = ['TS', 'CC', 'dlnA', r"$\chi$"]
        num_bins = [15, 15, 15, 15, 15, 15]
        vtype_dict = {'TS': "tshift", 'CC': "cc",
                      "power_l1_ratio(dB)": "power_l1",
                      "dlnA": "power_l2",
                      "CC amplitude ratio(dB)": "cc_amp",
                      r"$\chi$": "chi"}

        out_metric = []

        self.sort_metas()
        nrows = len(self.metas_sort.keys())
        ncols = len(vtype_list)

        plt.figure(figsize=(4*ncols, 4*nrows))
        G = gridspec.GridSpec(nrows, ncols)

        cat_names = sorted(self.metas_sort.keys())
        for irow, cat in enumerate(cat_names):
            self.plot_stats_histogram_one_category(
                G, irow, cat, vtype_list, num_bins, vtype_dict, out_metric, self.var_reduction)
        plt.tight_layout()
        plt.savefig(self.outputfn)
        with open("out_metric.json","w") as f:
            json.dump(out_metric,f)


class PlotInvSummary(object):

    def __init__(self, data_container=None, metas=None, cmtsource=None, config=None,
                 nregions=12, new_cmtsource=None, bootstrap_mean=None,
                 bootstrap_std=None, var_reduction=None, var_all=0.0, var_all_new=0.0, mode="regional"):
        self.data_container = data_container
        self.metas = metas
        self.cmtsource = cmtsource
        self.trwins = data_container.trwins
        self.config = config
        self.nregions = nregions

        self.new_cmtsource = new_cmtsource
        self.bootstrap_mean = bootstrap_mean
        self.bootstrap_std = bootstrap_std
        self.var_reduction = var_reduction
        self.var_all = var_all
        self.var_all_new = var_all_new

        if mode.lower() not in ["global", "regional"]:
            raise ValueError("Plot mode: 1) global; 2) regional")
        self.mode = mode.lower()

        self.sta_lat = None
        self.sta_lon = None
        self.sta_dist = []
        # azimuth in degree unit
        self.sta_azi = []
        # azimuth in radius unit
        self.sta_theta = []
        self.prepare_array()

    def prepare_array(self):
        # station
        self.sta_lat = [window.latitude for window in self.trwins]
        self.sta_lon = [window.longitude for window in self.trwins]

        for sta_lat, sta_lon in zip(self.sta_lat, self.sta_lon):
            dist, az, baz = gps2dist_azimuth(self.cmtsource.latitude,
                                             self.cmtsource.longitude,
                                             sta_lat, sta_lon)
            self.sta_azi.append(az)
            self.sta_theta.append(az / 180.0 * np.pi)
            if self.mode == "regional":
                # if regional, then use original distance(in km)
                self.sta_dist.append(dist / 1000.0)
            elif self.mode == "global":
                # if global, then use degree as unit
                self.sta_dist.append(dist/EARTH_HC)

    def get_azimuth_bin_number(self, azimuth):
        """
        Calculate the bin number of a given azimuth

        :param azimuth: test test test
        :return:
        """
        # the azimth ranges from [0,360]
        # so a little modification here
        daz = 360.0 / self.nregions
        k = int(math.floor(azimuth / daz))
        if k < 0 or k > self.nregions:
            if azimuth - 360.0 < 0.0001:
                k = self.nregions - 1
            else:
                raise ValueError('Error bining azimuth')
        return k

    def calculate_azimuth_bin(self, azimuth_array):
        """
        Calculate the azimuth and sort them into binsconda instal

        :return:
        """
        delta = 2*np.pi/self.nregions
        bins = [delta*i for i in range(self.nregions)]

        naz_wins = np.zeros(self.nregions)
        for azimuth in azimuth_array:
            bin_idx = self.get_azimuth_bin_number(azimuth[0])
            naz_wins[bin_idx] += azimuth[1]
        return bins, naz_wins

    @staticmethod
    def plot_si_bb(ax, cmt):
        # get moment tensor
        mt = [cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp]
        # plot beach ball
        b = beach(mt, linewidth=1, xy=(0, 0.6), width=1, size=2,facecolor='blue')
        ax.add_collection(b)
        # set axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1.5])
        ax.set_aspect('equal')
        # magnitude
        text = "Mw=%4.3f" % cmt.moment_magnitude
        plt.text(-0.9, -0.3, text, fontsize=7)
        # lat and lon
        text = "lat=%6.3f$^\\circ$; lon=%6.3f$^\\circ$" \
               % (cmt.latitude, cmt.longitude)
        plt.text(-0.9, -0.5, text, fontsize=7)
        # depth
        text = "dep=%6.3f km;" % (cmt.depth_in_m/1000.0)
        plt.text(-0.9, -0.7, text, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        # title
        text = "Init CMT"
        plt.text(-0.9, 1.3, text, fontsize=7)

    @staticmethod
    def plot_si_bb_comp(ax, cmt, cmt_init, tag):
        # get moment tensor
        mt = [cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp]
        # plot beach ball
        b = beach(mt, linewidth=1, xy=(0, 0.6), width=1, size=2,
                  facecolor='r')
        ax.add_collection(b)
        # set axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1.5])
        ax.set_aspect('equal')
        # magnitude
        text = r"$\Delta$Mw=%4.3f" % (
            cmt.moment_magnitude-cmt_init.moment_magnitude)
        plt.text(-0.9, -0.3, text, fontsize=7)
        # lat and lon
        text = r"$\Delta$lat=%6.3f$^\circ$; $\Delta$lon=%6.3f$^\circ$" \
               % (cmt.latitude-cmt_init.latitude,
                  cmt.longitude-cmt_init.longitude)
        plt.text(-0.9, -0.5, text, fontsize=7)
        # depth
        text = r"$\Delta$dep=%6.3f km;" % (
            (cmt.depth_in_m-cmt_init.depth_in_m)/1000.0)
        plt.text(-0.9, -0.7, text, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        text = tag
        plt.text(-0.9, 1.3, text, fontsize=7)

    def plot_table(self):
        par_mean = self.bootstrap_mean
        par_std = self.bootstrap_std
        std_over_mean = np.zeros(par_mean.shape)
        for _i in range(par_mean.shape[0]):
            if par_mean[_i] != 0:
                std_over_mean[_i] = par_std[_i]/np.abs(par_mean[_i])
            else:
                std_over_mean[_i] = 0.0
        fontsize = 9
        incre = 0.06
        pos = 1.00
        format1 = "    {:< 16.3e}   {:< 16.3e}   {:< 16.3e}   {:< 16.3e}   {:< 15.2%}"
        format2 = "    {:< 21.3f}  {:< 21.3f}  {:< 21.3f}  {:< 21.3f}  {:< 15.2%}"
        format3 = "    {:< 19.3f}  {:< 19.3f}  {:< 19.3f}  {:< 19.3f}  {:< 15.2%}"

        text = "Number of stations: %3d          Number of windows: %d" \
               % (len(self.sta_lat), self.data_container.nwindows) + \
               "         Envelope coef: %5.3f" % (self.config.envelope_coef)
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = "Number of Parameters: %3d        Zero-Trace: %6s" \
               "                  Double-couple: %6s " \
               % (self.config.npar, self.config.zero_trace,
                  self.config.double_couple)
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = "Station Correction: %6s         Norm_by_energy: %6s" \
               "         Norm_by_category: %6s" \
               % (self.config.station_correction,
                  self.config.weight_config.normalize_by_energy,
                  self.config.weight_config.normalize_by_category)
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        energy_change = \
            (self.new_cmtsource.M0 - self.cmtsource.M0) / self.cmtsource.M0
        text = "Inversion Damping: %6.3f       Energy Change: %6.2f%%" \
               "       Variance Reduction: %6.2f%%" \
               % (self.config.damping, energy_change*100,
                  self.var_reduction[1]*100)
        plt.text(0, pos, text, fontsize=fontsize)
        pos -= incre
        text = "Total Variance Reduced from %e to %e ===== %f %%" \
               % (self.var_all, self.var_all_new,
                  self.var_reduction[1]*100)
        plt.text(0, pos, text, fontsize=fontsize)


        pos -= incre
        text = "="*20 + "   Summary Table    " + "="*20
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = " PAR         Old_CMT          New_CMT       Bootstrap_Mean" \
               "      Bootstrap_STD     STD/Mean"
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = "Mrr: " + format1.format(
            self.cmtsource.m_rr, self.new_cmtsource.m_rr,
            par_mean[0], par_std[0], std_over_mean[0])
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mtt: " + format1.format(
            self.cmtsource.m_tt, self.new_cmtsource.m_tt,
            par_mean[1], par_std[1], std_over_mean[1])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mpp:" + format1.format(
            self.cmtsource.m_pp, self.new_cmtsource.m_pp,
            par_mean[2], par_std[2], std_over_mean[2])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mrt: " + format1.format(
            self.cmtsource.m_rt, self.new_cmtsource.m_rt,
            par_mean[3], par_std[3], std_over_mean[3])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mrp: " + format1.format(
            self.cmtsource.m_rp, self.new_cmtsource.m_rp,
            par_mean[4], par_std[4], std_over_mean[4])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mtp: " + format1.format(
            self.cmtsource.m_tp, self.new_cmtsource.m_tp,
            par_mean[5], par_std[5], std_over_mean[5])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "DEP:" + format3.format(
               self.cmtsource.depth_in_m,
               self.new_cmtsource.depth_in_m,
               par_mean[6], par_std[6], std_over_mean[6])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "LON:" + format2.format(
               self.cmtsource.longitude, self.new_cmtsource.longitude,
               par_mean[7], par_std[7], std_over_mean[7])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "LAT:" + format2.format(
               self.cmtsource.latitude, self.new_cmtsource.latitude,
               par_mean[8], par_std[8], std_over_mean[8])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "CMT:" + format2.format(
               self.cmtsource.time_shift, self.new_cmtsource.time_shift,
               par_mean[9], par_std[9], std_over_mean[9])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "HDR:" + format2.format(
               self.cmtsource.half_duration, self.new_cmtsource.half_duration,
               par_mean[10], par_std[10], std_over_mean[10])
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        plt.axis('off')

    def plot_global_map(self):
        """
        Plot global map of event and stations
        """

        # import basemap here due to error
        from mpl_toolkits.basemap import Basemap

        # ax = plt.subplot(211)
        plt.title(self.cmtsource.eventname)
        m = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0,
                    resolution='c')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 420., 60.))
        m.drawmapboundary()

        x, y = m(self.sta_lon, self.sta_lat)

        #print(self.sta_lon,self.sta_lat)
        #print(self.metas)
        #print(len(self.metas),len(self.sta_lon))
        #print(w)

        plt.scatter(x, y, s=30, c='red', cmap='afmhot', marker="^", edgecolor="k",
                  linewidth=0.3, zorder=3)

        cmt_lat = self.cmtsource.latitude
        cmt_lon = self.cmtsource.longitude
        focmecs = get_cmt_par(self.cmtsource)[:6]
        ax = plt.gca()
        if self.mode == 'regional':
            minlon = min(self.sta_lon)
            maxlon = max(self.sta_lon)
            minlat = min(self.sta_lat)
            maxlat = max(self.sta_lat)
            padding = 0.2
            m.drawparallels(np.arange(-90., 120., padding), labels=[1, 1, 0, 0], fmt="%.2f",
                dashes=[2, 2])
            m.drawmeridians(np.arange(0., 420., padding), labels=[0, 0, 1, 1], fmt="%.2f",
                dashes=[2, 2])
            m.etopo()
            ax.set_xlim(minlon-padding, maxlon+padding)
            ax.set_ylim(minlat-padding, maxlat+padding)
            width_beach = min((maxlon+2*padding-minlon)/(40*padding),
                              (maxlat+2*padding-minlat)/(40*padding))
        else:
            width_beach = 20
        bb = beach(focmecs, xy=(cmt_lon, cmt_lat),
                   width=width_beach, linewidth=1, alpha=1.0)
        bb.set_zorder(10)
        ax.add_collection(bb)

    def plot_global_map_weights(self):
        """
        Plot global map of event and stations
        """

        # import basemap here due to error
        from mpl_toolkits.basemap import Basemap

        # ax = plt.subplot(211)
        plt.title(self.cmtsource.eventname + "_Weights")
        m = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0,
                    resolution='c')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 420., 60.))
        m.drawmapboundary()

        #x, y = m(self.sta_lon, self.sta_lat)
        x, y = self.sta_lon, self.sta_lat
        w = [meta.weights for meta in self.metas]
        print(w)
        w /= max(w)
        w = map(float, w)
        #w = map(str, w)
        w = list(w)
        s = [np.sqrt(x)*100 for x in w]
        w = [x ** 0.5 for x in w]
        #print(self.sta_lon,self.sta_lat)
        #print(self.metas)
        #print(len(self.metas),len(self.sta_lon))
        #print(w)

        m.scatter(x, y, s=s, c=w, marker="^", cmap='hsv', zorder=3)
        cmt_lat = self.cmtsource.latitude
        cmt_lon = self.cmtsource.longitude
        focmecs = get_cmt_par(self.cmtsource)[:6]
        ax = plt.gca()
        if self.mode == 'regional':
            minlon = min(self.sta_lon)
            maxlon = max(self.sta_lon)
            minlat = min(self.sta_lat)
            maxlat = max(self.sta_lat)
            padding = 0.2
            m.drawparallels(np.arange(-90., 120., padding), labels=[1, 1, 0, 0], fmt="%.2f",
                dashes=[2, 2])
            m.drawmeridians(np.arange(0., 420., padding), labels=[0, 0, 1, 1], fmt="%.2f",
                dashes=[2, 2])
            m.etopo()
            ax.set_xlim(minlon-padding, maxlon+padding)
            ax.set_ylim(minlat-padding, maxlat+padding)
            width_beach = min((maxlon+2*padding-minlon)/(40*padding),
                              (maxlat+2*padding-minlat)/(40*padding))

        bb = beach(focmecs, xy=(cmt_lon, cmt_lat),
                   width=width_beach, linewidth=1, alpha=1.0)
        bb.set_zorder(10)
        ax.add_collection(bb)

    def plot_sta_dist_azi_weights(self):
        plt.title("", fontsize=10)
        ax = plt.gca()
        fig = plt.gcf()
        w = [meta.weights for meta in self.metas]
        w /= max(w)
        w = map(float, w)
        #w = map(str, w)
        w = list(w)
        s = [np.sqrt(x)*100 for x in w]
        w = [x ** 0.5 for x in w]
        c = plt.scatter(self.sta_theta, self.sta_dist, marker=u'^', c=w,
                        s=s, cmap='hsv')
        cbar = fig.colorbar(c, orientation='vertical', fraction=0.02, pad=0.15)
        plt.clim(min(w)/0.99, max(w)*0.99) 
        c.set_alpha(0.99)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        if self.mode == "regional":
            ax.set_rmax(1.10 * max(self.sta_dist))
        elif self.mode == "global":
            ax.set_rmax(1.0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_sta_dist_azi(self):
        plt.title("Station Dist and Azi", fontsize=10)
        ax = plt.gca()
        c = plt.scatter(self.sta_theta, self.sta_dist, marker=u'^', c='r',
                        s=20, edgecolor='k', linewidth=0.3)
        c.set_alpha(0.75)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        if self.mode == "regional":
            ax.set_rmax(1.10 * max(self.sta_dist))
        elif self.mode == "global":
            ax.set_rmax(1.0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_sta_azi(self):
        # set plt.subplot(***, polar=True)
        plt.title("Station Azimuth", fontsize=10)
        azimuth_array = []
        for azi in self.sta_azi:
            azimuth_array.append([azi, 1])
        bins, naz = self.calculate_azimuth_bin(azimuth_array)
        norm_factor = np.max(naz)
        print("bins=",bins)
        bars = plt.bar(bins, naz, width=(bins[1]-bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r/norm_factor))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_win_azi(self):
        # set plt.subplot(***, polar=True)
        plt.title("Window Azimuth", fontsize=10)
        win_azi = []
        for azi, window in zip(self.sta_azi, self.trwins):
            win_azi.append([azi, window.nwindows])
        bins, naz = self.calculate_azimuth_bin(win_azi)
        norm_factor = np.max(naz)

        bars = plt.bar(bins, naz, width=(bins[1]-bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r/norm_factor))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_dataset(self, figurename=None):
        """
        Plot only the dataset, including global map, station and window
        distribution, and beach ball
        """
        plt.figure(figsize=(10, 7), facecolor='w', edgecolor='k')
        g = gridspec.GridSpec(2, 3)
        plt.subplot(g[0, :-1])
        self.plot_global_map()
        plt.subplot(g[1, 0],  polar=True)
        self.plot_sta_dist_azi()
        plt.subplot(g[1, 1], polar=True)
        self.plot_sta_azi()
        plt.subplot(g[1, 2], polar=True)
        self.plot_win_azi()
        ax = plt.subplot(g[0, 2])
        self.plot_si_bb(ax, self.cmtsource)
        if figurename is None:
            plt.show()
        else:
            plt.savefig(figurename)

    def plot_inversion_summary(self, figurename=None):
        """
        Plot the dataset and the inversion result.
        """
        if self.new_cmtsource is None:
            raise ValueError("No new cmtsource...Can't plot summary")

        plt.figure(figsize=(10, 4), facecolor='w', edgecolor='k')
        g = gridspec.GridSpec(1, 3)
        plt.subplot(g[0, :-1])
        self.plot_global_map_weights()
        ax = plt.subplot(g[0, 2], polar=True)
        self.plot_sta_dist_azi_weights()
        if figurename is None:
            plt.show()
        else:
            #plt.show()
            plt.savefig("map_weights.png",dpi=300)

        plt.figure(figsize=(10, 11), facecolor='w', edgecolor='k')
        g = gridspec.GridSpec(3, 3)
        plt.subplot(g[0, :-1])
        self.plot_global_map()
        plt.subplot(g[1, 0],  polar=True)
        self.plot_sta_dist_azi()
        plt.subplot(g[1, 1], polar=True)
        self.plot_sta_azi()
        plt.subplot(g[1, 2], polar=True)
        self.plot_win_azi()
        ax = plt.subplot(g[0, 2])
        self.plot_si_bb(ax, self.cmtsource)
        ax = plt.subplot(g[2, 2])
        self.plot_si_bb_comp(ax, self.new_cmtsource, self.cmtsource,
                             "Inversion")
        plt.subplot(g[2, :-1])
        self.plot_table()
        if figurename is None:
            plt.show()
        else:
            plt.savefig(figurename)

