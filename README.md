# OAU Campus Transport Study

Field study of CNG bus and Keke services at Obafemi Awolowo University, Ile-Ife:
queues, waiting times (percentiles and survival curves), capacity, fleet availability,
a validated queue simulation of ways to cut the 90th-percentile wait, and a student survey.

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
