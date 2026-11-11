# OAU Campus Transport Study

Study of CNG bus and Keke services at Obafemi Awolowo University, Ile-Ife.
It combines **field records taken by hand by the team** (late September and 27 October 2026) with a
**model-estimated week** (26 Oct – 1 Nov 2026) that fills stops and hours not observed. The report keeps the
two apart, checks the model against the field records, and uses percentiles, survival curves and a queue
simulation to show where the longest waits come from and how to cut them.

**Live site:** add your Streamlit link here

## Files
| File | What it is |
|---|---|
| `app.py` | The whole website: report, dashboard, data and team pages (all code in one file) |
| `requirements.txt` | Python packages Streamlit installs |
| `team.csv` | Team names, roles and contributions (edit to update the credits) |
| `B1_Queue_and_Demand.csv` | Queue counts every 10 minutes |
| `B2_Wait_Sample.csv` | Timed passenger waits |
| `C_Vehicle_and_Trip.csv` | Every vehicle arrival and departure |
| `survey_clean.csv` | Cleaned student survey |
| `hypothesis_tests.csv`, `spearman_correlations.csv`, `survey_observed_wait_comparison.csv` | Statistical results |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Team
Akanji Sodiq Olamide · Salawu Aishat Olamide · Oyebanji Anuoluwapo Eniola · Adetomiwa Uthman Akinkunmi ·
Fagbohun Oluwatunmise Ayanfeoluwa · Adedeji Ibraheem Olalekan · Adekunle Dominion Oluwafemi · Ajagbe Samuel Oluwafikayomi.
All team members carried out the field work and data collection.
