# Data provenance

Creator: Kevin Hillstrom / MineThatData. Original challenge: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

Original published download:
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Downloaded 15 October 2026. The HTTPS version returned HTTP 502; the documented HTTP URL succeeded. Raw bytes: 3,964,977. SHA256: `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.

The dataset contains 64,000 rows and 12 columns, with no missing values. Assigned-arm counts: no email 21,306; men's email 21,307; women's email 21,387. These are acquisition checks, not model results. The publication describes random assignment and two-week follow-up.

Analysis uses the publicly distributed challenge. An express raw-data redistribution license has not been established, so raw data and customer-level predictions are excluded from the repository and its code license. Readers obtain data from the provider/maintained loader under applicable terms. Derived aggregate results cite the original source. No claim of ownership of the dataset is made.

Maintainer cross-check: scikit-uplift's documented loader provides an HTTPS mirror and checksum:
https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html#fetch_hillstrom

No demographic gender is present: `mens` and `womens` indicate prior purchase categories. Do not relabel them gender.

The maintained HTTPS gzip matched its published MD5 `a68a81291f53a14f4e29002629803ba3`; parsed cells were compared with the original CSV and matched exactly. The loader prefers that HTTPS mirror for new downloads. Binary, numeric-range, treatment-level and outcome-consistency checks passed; all 64,000 rows are retained. The data audit records independent split sizes and row-position hashes.
