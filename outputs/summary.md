# SAFIRI — Interpretable Anomaly Triage Report

This system ranks customs/shipment records for investigation with limited review resources.
Training-only fitting, validation-based threshold calibration, and frozen-threshold evaluation on test and synthetic stress tests.

## Output Components (per record)

| Component | Meaning |
|-----------|---------|
| **risk_score** | Investigation priority (anomaly evidence × business exposure) |
| **anomaly_score** | Pure deviation from normal behaviour (peer + unsupervised) |
| **confidence_score** | Evidence reliability (peer group size, tier agreement, data completeness) |
| **supervised_score** | XGBoost fraud probability (Tier 4, supervised learning) |
| **explanation** | Human-readable reason tied to peer context or rule violation |

## Pipeline Configuration

- Anomaly evidence formula: `evidence = 0.03*rule + 0.04*peer + 0.03*IF + 0.90*supervised`
- Risk formula: `risk = evidence × (1 + 0.10 × business_exposure)`
- LOF enabled: `True`
- Supervised (XGBoost) enabled: `True`
- Deep learning (MLP) enabled: `False`
- Validation-calibrated flag threshold on `risk_score`: `0.836224`
- Review capacity target: top `5.00%` of validation cases

## Validation Risk Ranking

- `average_precision`: 0.5797
- `precision_at_1pct`: 0.8293
- `precision_at_3pct`: 0.8531
- `precision_at_5pct`: 0.8231
- `precision_at_10pct`: 0.7248
- `recall_at_5pct`: 0.1966
- `recall_at_10pct`: 0.3462
- `recall_at_20pct`: 0.5346
- `roc_auc`: 0.7969

## Validation Risk Threshold

- `accuracy`: 0.8228
- `precision`: 0.8231
- `recall`: 0.1966
- `f1`: 0.3174
- `flag_rate`: 0.0500

## Validation Anomaly Ranking

- `average_precision`: 0.2018
- `precision_at_1pct`: 0.2439
- `precision_at_3pct`: 0.1878
- `precision_at_5pct`: 0.1867
- `precision_at_10pct`: 0.2015
- `recall_at_5pct`: 0.0446
- `recall_at_10pct`: 0.0962
- `recall_at_20pct`: 0.1831
- `roc_auc`: 0.4812

## Validation Anomaly Threshold

- `accuracy`: 0.8228
- `precision`: 0.8231
- `recall`: 0.1966
- `f1`: 0.3174
- `flag_rate`: 0.0500

## Test Risk Ranking

- `average_precision`: 0.5700
- `precision_at_1pct`: 0.8000
- `precision_at_3pct`: 0.8000
- `precision_at_5pct`: 0.7647
- `precision_at_10pct`: 0.7091
- `recall_at_5pct`: 0.1771
- `recall_at_10pct`: 0.3281
- `recall_at_20pct`: 0.5379
- `roc_auc`: 0.7933

## Test Risk Threshold

- `accuracy`: 0.8093
- `precision`: 0.7685
- `recall`: 0.1700
- `f1`: 0.2784
- `flag_rate`: 0.0479

## Test Anomaly Ranking

- `average_precision`: 0.2061
- `precision_at_1pct`: 0.1765
- `precision_at_3pct`: 0.1804
- `precision_at_5pct`: 0.1835
- `precision_at_10pct`: 0.1991
- `recall_at_5pct`: 0.0425
- `recall_at_10pct`: 0.0921
- `recall_at_20pct`: 0.1869
- `roc_auc`: 0.4795

## Test Anomaly Threshold

- `accuracy`: 0.8093
- `precision`: 0.7685
- `recall`: 0.1700
- `f1`: 0.2784
- `flag_rate`: 0.0479

## Synthetic Risk Ranking

- `average_precision`: 0.0293
- `precision_at_1pct`: 0.0238
- `precision_at_3pct`: 0.0317
- `precision_at_5pct`: 0.0286
- `precision_at_10pct`: 0.0274
- `recall_at_5pct`: 0.0500
- `recall_at_10pct`: 0.0958
- `recall_at_20pct`: 0.2125
- `roc_auc`: 0.5130

## Synthetic Risk Threshold

- `accuracy`: 0.9238
- `precision`: 0.0284
- `recall`: 0.0500
- `f1`: 0.0363
- `flag_rate`: 0.0504

## Synthetic Anomaly Ranking

- `average_precision`: 0.0877
- `precision_at_1pct`: 0.2262
- `precision_at_3pct`: 0.1508
- `precision_at_5pct`: 0.1360
- `precision_at_10pct`: 0.0919
- `recall_at_5pct`: 0.2375
- `recall_at_10pct`: 0.3208
- `recall_at_20pct`: 0.4375
- `roc_auc`: 0.6630

## Synthetic Anomaly Threshold

- `accuracy`: 0.9238
- `precision`: 0.0284
- `recall`: 0.0500
- `f1`: 0.0363
- `flag_rate`: 0.0504

## Synthetic Revenue At 5Pct

- `synthetic_revenue_at_5pct`: 198727345.2279

## Synthetic Breakdown

### arithmetic_mismatch

- `count`: 48
- `recall`: 0.0000
- `avg_risk_score`: 0.2723
- `avg_anomaly_score`: 0.7216

### rare_pair_swap

- `count`: 48
- `recall`: 0.0625
- `avg_risk_score`: 0.3823
- `avg_anomaly_score`: 0.5873

### repeated_rounded_value

- `count`: 48
- `recall`: 0.0625
- `avg_risk_score`: 0.3121
- `avg_anomaly_score`: 0.7942

### tax_indicator_mismatch

- `count`: 48
- `recall`: 0.0625
- `avg_risk_score`: 0.3325
- `avg_anomaly_score`: 0.6061

### valuation_peer_shift

- `count`: 48
- `recall`: 0.0625
- `avg_risk_score`: 0.3275
- `avg_anomaly_score`: 0.8467


## Explanation Quality

### validation

- `flagged_records`: 407
- `coverage_pct`: 100.0000

### test

- `flagged_records`: 406
- `coverage_pct`: 100.0000

### synthetic

- `flagged_records`: 422
- `coverage_pct`: 100.0000


## Tier Diagnostics

### validation

#### tier1_rule

- `average_precision`: 0.2012
- `precision_at_1pct`: 0.0854
- `precision_at_3pct`: 0.1796
- `precision_at_5pct`: 0.1990
- `precision_at_10pct`: 0.2039
- `recall_at_5pct`: 0.0475
- `recall_at_10pct`: 0.0974
- `recall_at_20pct`: 0.1743
- `roc_auc`: 0.4849

#### tier2_peer

- `average_precision`: 0.2005
- `precision_at_1pct`: 0.1829
- `precision_at_3pct`: 0.2000
- `precision_at_5pct`: 0.2138
- `precision_at_10pct`: 0.1892
- `recall_at_5pct`: 0.0511
- `recall_at_10pct`: 0.0904
- `recall_at_20pct`: 0.1860
- `roc_auc`: 0.4812

#### tier3_if

- `average_precision`: 0.2029
- `precision_at_1pct`: 0.1707
- `precision_at_3pct`: 0.2000
- `precision_at_5pct`: 0.1843
- `precision_at_10pct`: 0.1867
- `recall_at_5pct`: 0.0440
- `recall_at_10pct`: 0.0892
- `recall_at_20pct`: 0.1942
- `roc_auc`: 0.4883

#### tier4_supervised

- `average_precision`: 0.5895
- `precision_at_1pct`: 0.8780
- `precision_at_3pct`: 0.8531
- `precision_at_5pct`: 0.8452
- `precision_at_10pct`: 0.7273
- `recall_at_5pct`: 0.2019
- `recall_at_10pct`: 0.3474
- `recall_at_20pct`: 0.5358
- `roc_auc`: 0.8006


### test

#### tier1_rule

- `average_precision`: 0.2113
- `precision_at_1pct`: 0.2588
- `precision_at_3pct`: 0.2078
- `precision_at_5pct`: 0.2165
- `precision_at_10pct`: 0.2108
- `recall_at_5pct`: 0.0501
- `recall_at_10pct`: 0.0975
- `recall_at_20pct`: 0.1820
- `roc_auc`: 0.4877

#### tier2_peer

- `average_precision`: 0.2096
- `precision_at_1pct`: 0.2824
- `precision_at_3pct`: 0.2471
- `precision_at_5pct`: 0.2141
- `precision_at_10pct`: 0.2014
- `recall_at_5pct`: 0.0496
- `recall_at_10pct`: 0.0932
- `recall_at_20pct`: 0.1847
- `roc_auc`: 0.4810

#### tier3_if

- `average_precision`: 0.2139
- `precision_at_1pct`: 0.2706
- `precision_at_3pct`: 0.2314
- `precision_at_5pct`: 0.2094
- `precision_at_10pct`: 0.2250
- `recall_at_5pct`: 0.0485
- `recall_at_10pct`: 0.1041
- `recall_at_20pct`: 0.2000
- `roc_auc`: 0.4870

#### tier4_supervised

- `average_precision`: 0.5744
- `precision_at_1pct`: 0.8235
- `precision_at_3pct`: 0.8118
- `precision_at_5pct`: 0.7812
- `precision_at_10pct`: 0.7079
- `recall_at_5pct`: 0.1809
- `recall_at_10pct`: 0.3275
- `recall_at_20pct`: 0.5357
- `roc_auc`: 0.7972


### synthetic

#### tier1_rule

- `average_precision`: 0.1752
- `precision_at_1pct`: 0.2619
- `precision_at_3pct`: 0.2937
- `precision_at_5pct`: 0.2100
- `precision_at_10pct`: 0.1169
- `recall_at_5pct`: 0.3667
- `recall_at_10pct`: 0.4083
- `recall_at_20pct`: 0.6083
- `roc_auc`: 0.7559

#### tier2_peer

- `average_precision`: 0.0913
- `precision_at_1pct`: 0.2143
- `precision_at_3pct`: 0.1468
- `precision_at_5pct`: 0.1384
- `precision_at_10pct`: 0.0943
- `recall_at_5pct`: 0.2417
- `recall_at_10pct`: 0.3292
- `recall_at_20pct`: 0.4875
- `roc_auc`: 0.6668

#### tier3_if

- `average_precision`: 0.0498
- `precision_at_1pct`: 0.0357
- `precision_at_3pct`: 0.0754
- `precision_at_5pct`: 0.0811
- `precision_at_10pct`: 0.0668
- `recall_at_5pct`: 0.1417
- `recall_at_10pct`: 0.2333
- `recall_at_20pct`: 0.3792
- `roc_auc`: 0.6263

#### tier4_supervised

- `average_precision`: 0.0283
- `precision_at_1pct`: 0.0238
- `precision_at_3pct`: 0.0238
- `precision_at_5pct`: 0.0263
- `precision_at_10pct`: 0.0251
- `recall_at_5pct`: 0.0458
- `recall_at_10pct`: 0.0875
- `recall_at_20pct`: 0.1958
- `roc_auc`: 0.4909



## Top Validation Cases

|   declaration_id |   risk_score |   anomaly_score |   confidence_score |   business_exposure |   supervised_score | risk_tier   | flagged   | explanation                                                                                                                                                                                                                                              |
|-----------------:|-------------:|----------------:|-------------------:|--------------------:|-------------------:|:------------|:----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|         17853069 |       1      |          0.8897 |             0.8    |              0.6695 |             0.9846 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; overall feature profile is isolated from historical shipment patterns                                                                                                         |
|         99488413 |       1      |          0.6541 |             0.725  |              0.9909 |             0.9596 | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a suspiciously round value                                                                                                                 |
|         53367207 |       1      |          0.8483 |             0.6771 |              0.9609 |             0.9672 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.1% of peer group 'hs6' (n=35); value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline                                         |
|         75967717 |       1      |          0.6857 |             0.725  |              0.9632 |             0.9699 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; tax-type and origin-indicator combination is almost never seen in training history; high-value shipment with elevated tax rate — prioritised for investigation                |
|         64932059 |       1      |          0.8857 |             0.8    |              0.5435 |             0.9853 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.5% of peer group 'global' (n=37385); value deviates sharply from the importer's historical baseline; overall feature profile is isolated from historical shipment patterns                               |
|         92780105 |       1      |          0.8665 |             0.8    |              0.967  |             0.9658 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.6% of peer group 'global' (n=37385); valuation is 507.0× above the peer median (group 'global'); value deviates sharply from the importer's historical baseline                                          |
|         54926085 |       1      |          0.487  |             0.6749 |              0.9668 |             0.989  | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records                                                                                     |
|         54612307 |       1      |          0.7126 |             0.6021 |              0.9649 |             0.9625 | High Risk   | True      | High Risk: value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation; overall feature profile is isolated from historical shipment patterns                          |
|         57636694 |       1      |          0.9353 |             0.8    |              0.8617 |             0.9685 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.7% of peer group 'global' (n=37385); valuation is 364.2× above the peer median (group 'global'); value deviates sharply from the importer's historical baseline                                          |
|         66330983 |       0.9977 |          0.3261 |             0.5421 |              0.8759 |             0.9933 | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation                                                                                                                                                                    |
|         96658967 |       0.9973 |          0.9546 |             0.8    |              0.5435 |             0.9761 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.5% of peer group 'global' (n=37385); value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline                                   |
|         22849717 |       0.9945 |          0.257  |             0.6749 |              0.9668 |             0.9828 | High Risk   | True      | High Risk: tax-type and origin-indicator combination is almost never seen in training history; high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records |
|         60698172 |       0.9925 |          0.6887 |             0.725  |              0.81   |             0.9652 | High Risk   | True      | High Risk: declared item price is a suspiciously round value; moderate business exposure amplifies investigation priority                                                                                                                                |
|         31026714 |       0.9911 |          0.1532 |             0.6749 |              0.9668 |             0.9877 | High Risk   | True      | High Risk: tax-type and origin-indicator combination is almost never seen in training history; high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records |
|         13009874 |       0.9843 |          0.7264 |             0.8    |              0.81   |             0.9536 | High Risk   | True      | High Risk: declared item price is a suspiciously round value; moderate business exposure amplifies investigation priority                                                                                                                                |
|         66444252 |       0.9834 |          0.3306 |             0.5421 |              0.8759 |             0.9776 | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation                                                                                                                                                                    |
|         87109452 |       0.9828 |          0.5276 |             0.8021 |              0.9906 |             0.9501 | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation; overall feature profile is isolated from historical shipment patterns; declared item price is a suspiciously round value                                          |
|         90897553 |       0.9811 |          0.6553 |             0.725  |              0.5683 |             0.9812 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline                                                                                                                                                                                |
|         35774043 |       0.9802 |          0.7371 |             0.7228 |              0.7709 |             0.953  | High Risk   | True      | High Risk: overall feature profile is isolated from historical shipment patterns; moderate business exposure amplifies investigation priority                                                                                                            |
|         87173340 |       0.9792 |          0.9234 |             0.8    |              0.7757 |             0.9374 | High Risk   | True      | High Risk: valuation sits in the most extreme 2.1% of peer group 'global' (n=37385); value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline                                   |

## Top Test Cases

|   declaration_id |   risk_score |   anomaly_score |   confidence_score |   business_exposure |   supervised_score | risk_tier   | flagged   | explanation                                                                                                                                                                                                                                              |
|-----------------:|-------------:|----------------:|-------------------:|--------------------:|-------------------:|:------------|:----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|         78704306 |       1      |          0.8714 |             0.7285 |              0.7685 |             0.9824 | High Risk   | True      | High Risk: valuation is 3.3× below the peer median (group 'hs6'); value deviates sharply from the importer's historical baseline; tax-type and origin-indicator combination is almost never seen in training history                                     |
|         62785249 |       1      |          0.4353 |             0.6749 |              0.9668 |             0.9939 | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records                                                                                     |
|         31433636 |       1      |          0.6842 |             0.7705 |              0.861  |             0.9751 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; high-value shipment with elevated tax rate — prioritised for investigation                                                                                                    |
|         93401815 |       1      |          0.7868 |             0.725  |              0.828  |             0.9732 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; moderate business exposure amplifies investigation priority                                                |
|         29621117 |       1      |          0.7421 |             0.7499 |              0.9997 |             0.9624 | High Risk   | True      | High Risk: value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records                  |
|         60033126 |       1      |          0.8264 |             0.8455 |              0.861  |             0.9598 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation                                 |
|         54386717 |       1      |          0.7312 |             0.8282 |              0.859  |             0.9913 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation                                 |
|         15272454 |       1      |          0.7499 |             0.7246 |              0.9997 |             0.982  | High Risk   | True      | High Risk: valuation sits in the most extreme 0.1% of peer group 'hs6_origin_import' (n=32); value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation               |
|         54427860 |       1      |          0.5295 |             0.725  |              0.9542 |             0.9742 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a suspiciously round value                                                 |
|         17075336 |       1      |          0.4867 |             0.725  |              0.8078 |             0.9908 | High Risk   | True      | High Risk: declared item price is a suspiciously round value; moderate business exposure amplifies investigation priority                                                                                                                                |
|         73629967 |       0.9994 |          0.8577 |             0.8455 |              0.861  |             0.9549 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation                                 |
|         73678076 |       0.9987 |          0.7836 |             0.5949 |              0.8465 |             0.9628 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.1% of peer group 'hs6' (n=27); valuation is 13.6× above the peer median (group 'hs6'); value deviates sharply from the importer's historical baseline                                                    |
|         68671321 |       0.9952 |          0.9143 |             0.8    |              0.5617 |             0.9753 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.9% of peer group 'global' (n=37385); value deviates sharply from the importer's historical baseline; overall feature profile is isolated from historical shipment patterns                               |
|         93726240 |       0.9928 |          0.5022 |             0.7499 |              0.9668 |             0.9629 | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records; overall feature profile is isolated from historical shipment patterns              |
|         10390987 |       0.9927 |          0.4956 |             0.725  |              0.9542 |             0.968  | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a suspiciously round value                                                                                                                 |
|         62882900 |       0.9925 |          0.7761 |             0.7324 |              0.6127 |             0.9782 | High Risk   | True      | High Risk: valuation sits in the most extreme 0.1% of peer group 'hs6_origin' (n=27); value deviates sharply from the importer's historical baseline; overall feature profile is isolated from historical shipment patterns                              |
|         69369759 |       0.9917 |          0.3314 |             0.6749 |              0.9668 |             0.9741 | High Risk   | True      | High Risk: tax-type and origin-indicator combination is almost never seen in training history; high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records |
|         86919208 |       0.9904 |          0.8849 |             0.8249 |              0.9668 |             0.9315 | High Risk   | True      | High Risk: value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation; declared item price is a repeated rounded value across many unrelated records                  |
|         25530811 |       0.9903 |          0.7874 |             0.725  |              0.7068 |             0.9668 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; moderate business exposure amplifies investigation priority                                                |
|         55484092 |       0.9898 |          0.7495 |             0.7705 |              0.861  |             0.9543 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; high-value shipment with elevated tax rate — prioritised for investigation                                 |

## Top Synthetic Anomalies

|   declaration_id |   risk_score |   anomaly_score |   confidence_score |   business_exposure |   supervised_score | risk_tier   | flagged   | explanation                                                                                                                                                                                                                           |
|-----------------:|-------------:|----------------:|-------------------:|--------------------:|-------------------:|:------------|:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        900000028 |       0.9871 |          0.6559 |             0.725  |              0.5666 |             0.986  | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; overall feature profile is isolated from historical shipment patterns                                                                                      |
|        900000041 |       0.9396 |          0.6342 |             0.5939 |              0.4653 |             0.95   | High Risk   | True      | High Risk: valuation sits in the most extreme 0.1% of peer group 'hs6' (n=26); valuation is 7.3× above the peer median (group 'hs6')                                                                                                  |
|        900000202 |       0.913  |          0.726  |             0.7988 |              0.4376 |             0.9148 | High Risk   | True      | High Risk: overall feature profile is isolated from historical shipment patterns                                                                                                                                                      |
|        900000134 |       0.9102 |          0.9188 |             0.7882 |              0.3047 |             0.9093 | High Risk   | True      | High Risk: value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; overall feature profile is isolated from historical shipment patterns                   |
|        900000104 |       0.895  |          0.3832 |             0.6005 |              0.8521 |             0.8849 | High Risk   | True      | High Risk: high-value shipment with elevated tax rate — prioritised for investigation; overall feature profile is isolated from historical shipment patterns                                                                          |
|        900000031 |       0.8925 |          0.9781 |             0.8347 |              0.3231 |             0.884  | High Risk   | True      | High Risk: valuation sits in the most extreme 0.1% of peer group 'hs6_origin_import' (n=112); valuation is 12.6× below the peer median (group 'hs6_origin_import'); value deviates sharply from the importer's historical baseline    |
|        900000204 |       0.8908 |          0.3959 |             0.65   |              0.7694 |             0.8866 | High Risk   | True      | High Risk: tax-type and origin-indicator combination is almost never seen in training history; moderate business exposure amplifies investigation priority                                                                            |
|        900000235 |       0.8831 |          0.4176 |             0.7308 |              0.2675 |             0.9194 | High Risk   | True      | High Risk: tax-type and origin-indicator combination is almost never seen in training history; overall feature profile is isolated from historical shipment patterns                                                                  |
|        900000155 |       0.8816 |          0.5111 |             0.65   |              0.5636 |             0.8853 | High Risk   | True      | High Risk: declared item price is a repeated rounded value across many unrelated records                                                                                                                                              |
|        900000146 |       0.842  |          0.9472 |             0.8738 |              0.4216 |             0.821  | Medium Risk | True      | Medium Risk: valuation sits in the most extreme 1.6% of peer group 'hs6_origin_import' (n=445); valuation is 20.8× above the peer median (group 'hs6_origin_import'); value deviates sharply from the importer's historical baseline  |
|        900000108 |       0.8419 |          0.3208 |             0.5209 |              0.1136 |             0.897  | Medium Risk | True      | Medium Risk: tax-type and origin-indicator combination is logically contradictory                                                                                                                                                     |
|        900000160 |       0.8381 |          0.1432 |             0.65   |              0.5855 |             0.8651 | Medium Risk | True      | Medium Risk: declared item price is a repeated rounded value across many unrelated records                                                                                                                                            |
|        900000163 |       0.8298 |          0.9387 |             0.823  |              0.7146 |             0.785  | Normal      | False     | valuation sits in the most extreme 0.1% of peer group 'hs6_origin_import' (n=74); valuation is 65.5× above the peer median (group 'hs6_origin_import'); declared item price is a repeated rounded value across many unrelated records |
|        900000125 |       0.8137 |          0.9649 |             0.7303 |              0.2422 |             0.8074 | Normal      | False     | valuation sits in the most extreme 0.1% of peer group 'hs6_origin' (n=25); valuation is 5.0× below the peer median (group 'hs6_origin'); value deviates sharply from the seller-origin historical baseline                            |
|        900000237 |       0.802  |          0.4665 |             0.8021 |              0.9906 |             0.7703 | Normal      | False     | tax-type and origin-indicator combination is almost never seen in training history; high-value shipment with elevated tax rate — prioritised for investigation; overall feature profile is isolated from historical shipment patterns |
|        900000112 |       0.7953 |          0.9178 |             0.7424 |              0.0649 |             0.8059 | Normal      | False     | value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; overall feature profile is isolated from historical shipment patterns                              |
|        900000091 |       0.7952 |          0.8962 |             0.7016 |              0.3054 |             0.7821 | Normal      | False     | declared total deviates by 56.6% from quantity × unit_price; valuation sits in the most extreme 2.4% of peer group 'hs6' (n=84); valuation is 1.6× above the peer median (group 'hs6')                                                |
|        900000144 |       0.7949 |          0.7395 |             0.6393 |              0.5155 |             0.7814 | Normal      | False     | valuation sits in the most extreme 0.1% of peer group 'hs6' (n=132); valuation is 325.5× above the peer median (group 'hs6'); tax-type and origin-indicator combination is logically contradictory                                    |
|        900000025 |       0.7727 |          0.9481 |             0.8    |              0.7204 |             0.7265 | Normal      | False     | value deviates sharply from the importer's historical baseline; value deviates sharply from the seller-origin historical baseline; overall feature profile is isolated from historical shipment patterns                              |
|        900000218 |       0.7713 |          0.3974 |             0.65   |              0.4296 |             0.7921 | Normal      | False     | moderately unusual combined anomaly profile                                                                                                                                                                                           |
