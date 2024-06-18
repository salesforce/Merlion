#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the running file that implements windstats with both weekly and monthly seasonalities.
For the implementation of only weekly/monthly seasonality, specify "enable_weekly" of "enable_monthly" arguments of RunWindStats().
"""

from merlion.models.anomaly.windstats import WindStats, WindStatsConfig
from merlion.models.anomaly.windstats_monthly import MonthlyWindStats, MonthlyWindStatsConfig
from merlion.utils import TimeSeries

class RunWindStats:
    def __init__(self, threshold, enable_weekly = True, enable_monthly = True, post_rule_on_anom_score = False, WeeklyWindStatsConfig = WindStatsConfig(), MonthlyWindStatsConfig = MonthlyWindStatsConfig()):
        """
        Users can customize the configuration for weekly or monthly-based windstats. If not, then the default configuration will apply.
        """
                
        self.enable_weekly = enable_weekly
        self.enable_monthly = enable_monthly
        assert self.enable_weekly == True or self.enable_monthly == True, "Must enable either weekly or monthly seasonality, or both!"
        
        # Threshold on identifying anomaly based on anomaly score.
        self.threshold = threshold
        # If apply post rules on anomaly score
        self.post_rule = post_rule_on_anom_score
        
        # Intialize according model if enable weekly/monthly analysis
        if self.enable_weekly:
            self.model_weekly  = WindStats(WeeklyWindStatsConfig)
        if self.enable_monthly:
            self.model_monthly = MonthlyWindStats(MonthlyWindStatsConfig)

    # Identify anomaly based on the hard threshold.
    def anomalyByScore(self, scores, threshold):
        labels = scores.copy()
        labels.loc[abs(labels["anom_score"]) <= threshold] = 0
        labels.loc[abs(labels["anom_score"]) > threshold] = 1
        
        labels.rename(columns = {"anom_score": "anomaly"}, inplace = True)
        return labels
    
    # Filter anomaly scores based on post rules. Same as "get_anomaly_label" in WindStats
    def get_anomaly_label(self, model, ts):
        scores = model.train(ts)
        return model.post_rule(scores) if model.post_rule is not None else scores
    
    def run(self, ts):
        if self.enable_weekly:
            if self.post_rule:
                scores_weekly = self.get_anomaly_label(self.model_weekly, ts).to_pd()
            else:
                scores_weekly = self.model_weekly.train(ts).to_pd()
            labels_weekly = self.anomalyByScore(scores_weekly, self.threshold)
        
        if self.enable_monthly:
            if self.post_rule:
                scores_monthly = self.get_anomaly_label(self.model_monthly, ts).to_pd()
            else:
                scores_monthly = self.model_monthly.train(ts).to_pd()
            labels_monthly = self.anomalyByScore(scores_monthly, self.threshold)
            
        # Anomaly is identified if and only if it's detected in both weekly and monthly patterns.
        if self.enable_weekly and self.enable_monthly:
            return scores_weekly, scores_monthly, labels_weekly * labels_monthly
        elif self.enable_weekly:
            return scores_weekly, None, labels_weekly
        else:
            return None, scores_monthly, labels_monthly
