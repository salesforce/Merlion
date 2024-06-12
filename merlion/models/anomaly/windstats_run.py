#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the running file that implements windstats with both weekly and monthly seasonalities.
For the implementation of only weekly/monthly seasonality, specify "enable_weekly" of "enable_monthly" arguments of RunWindStats().
"""

from windstats import WindStats, WindStatsConfig
from windstats_monthly import MonthlyWindStats, MonthlyWindStatsConfig
from ts_datasets.anomaly import NAB
from merlion.utils import TimeSeries
from merlion.post_process.threshold import AggregateAlarms

class RunWindStats:
    def __init__(self, threshold, enable_weekly = True, enable_monthly = True, WeeklyWindStatsConfig = WindStatsConfig(), MonthlyWindStatsConfig = MonthlyWindStatsConfig()):
        """
        Users can customize the configuration for weekly or monthly-based windstats. If not, then the default configuration will apply.
        """
                
        self.enable_weekly = enable_weekly
        self.enable_monthly = enable_monthly
        assert self.enable_weekly == True or self.enable_monthly == True, "Must enable either weekly or monthly seasonality, or both!"
        
        # Threshold on identifying anomaly based on anomaly score.
        self.threshold = threshold
        
        if self.enable_weekly:
            self.model_weekly  = WindStats(WeeklyWindStatsConfig)
            
        if self.enable_monthly:
            self.model_monthly = MonthlyWindStats(MonthlyWindStatsConfig)

    def anomalyByScore(self, scores, threshold):
        scores.loc[abs(scores["anom_score"]) <= threshold] = 0
        scores.loc[abs(scores["anom_score"]) > threshold] = 1
        
        scores.rename(columns = {"anom_score": "anomaly"}, inplace = True)
        return scores
    
    def run(self, ts):
        if self.enable_weekly:
            scores_weekly = self.model_weekly.train(ts).to_pd()
            scores_weekly = self.anomalyByScore(scores_weekly, self.threshold)
        
        if self.enable_monthly:
            scores_monthly = self.model_monthly.train(ts).to_pd()
            scores_monthly = self.anomalyByScore(scores_monthly, self.threshold)
            
        if self.enable_weekly and self.enable_monthly:
            return scores_weekly * scores_monthly
        elif self.enable_weekly:
            return scores_weekly
        else:
            return scores_monthly
