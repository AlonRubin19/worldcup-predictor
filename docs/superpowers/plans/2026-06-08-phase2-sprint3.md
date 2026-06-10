# Phase 2 Sprint 3 Implementation Plan — Player Impact Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete player/squad impact pipeline that modifies xG based on expected starting XI strength and player availability, clearly labelled as placeholder until real data is sourced.

**Architecture:** Two new CSVs → player_loader.py → player_impact.py (squad_factor formula) → player_impact_runner.py (MLE + DC + squad adjustment). All existing runners/models are untouched. One comparison report compares MLE+DC vs MLE+DC+Player Impact.

**Tech Stack:** Python 3.10+, pandas, pytest (no new dependencies)

---

## Critical Context

- **130 tests passing** at Sprint 3 start.
- WC 2022 teams: 32 teams (see `data/historical_matches.csv`)
- Match IDs: `historical_matches.csv` rows are 0-indexed in pandas but `match_id` in our data is 1-indexed.
- The `match_results.csv` uses a bidirectional resolver (from Sprint 2) to align team ordering.
- Benzema (France) withdrew injured before the tournament — good demo case for player impact.
- Notable WC 2022 availability events:
  - France: Benzema withdrew before match 1 (injury)
  - Senegal: Sadio Mané withdrew before the tournament (injured)
  - Germany: Neuer was fit for all group matches

## File Map

| File | Action |
|---|---|
| `data/player_profiles.csv` | Create — ~400 rows, 32 teams |
| `data/match_player_availability.csv` | Create — ~17,600 rows (40 matches × 32 teams × ~14 players avg) |
| `docs/player_data_status.md` | Create — sourcing requirements |
| `src/data/player_loader.py` | Create |
| `src/models/player_impact.py` | Create |
| `src/backtesting/player_impact_runner.py` | Create |
| `tests/data/test_player_loader.py` | Create — 6 tests |
| `tests/models/test_player_impact.py` | Create — 8 tests |
| `tests/backtesting/test_player_impact_runner.py` | Create — 5 tests |
| `scripts/run_sprint3_report.py` | Create — comparison report |

---

## Task 1: Player Profiles Dataset

**Files:**
- Create: `data/player_profiles.csv`
- Create: `docs/player_data_status.md`

- [ ] **Step 1: Create `data/player_profiles.csv`**

```
# WARNING: PLACEHOLDER DATA — not sourced from real player records.
# Engineering validation only. See docs/player_data_status.md for sourcing requirements.
# base_impact_score: composite 0.0-1.0 (1.0 = world-class)
# positions: GK=goalkeeper, DEF=defender, MID=midfielder, FWD=forward
player_id,player_name,team,position,club,minutes_last_90_days,national_team_minutes_last_12_months,goals_per_90,assists_per_90,xg_per_90,xa_per_90,defensive_actions_per_90,international_caps,base_impact_score
ARG_001,Lionel Messi,Argentina,FWD,Paris Saint-Germain,2100,900,0.65,0.55,0.72,0.60,3.2,172,0.95
ARG_002,Julián Álvarez,Argentina,FWD,Manchester City,1800,720,0.55,0.25,0.60,0.28,4.1,18,0.72
ARG_003,Ángel Di María,Argentina,MID,Juventus,1650,600,0.28,0.35,0.32,0.40,3.5,129,0.70
ARG_004,Rodrigo De Paul,Argentina,MID,Atletico Madrid,2200,810,0.12,0.22,0.14,0.24,7.8,58,0.65
ARG_005,Nicolás Otamendi,Argentina,DEF,Benfica,2100,720,0.06,0.03,0.07,0.04,9.2,97,0.62
ARG_006,Cristian Romero,Argentina,DEF,Tottenham,1900,680,0.05,0.04,0.06,0.05,10.1,28,0.65
ARG_007,Alexis Mac Allister,Argentina,MID,Brighton,2000,650,0.18,0.15,0.20,0.16,6.2,17,0.63
ARG_008,Leandro Paredes,Argentina,MID,Juventus,1700,580,0.05,0.12,0.06,0.14,6.8,50,0.58
ARG_009,Emiliano Martínez,Argentina,GK,Aston Villa,2700,900,0.00,0.00,0.00,0.00,1.2,21,0.72
ARG_010,Nahuel Molina,Argentina,DEF,Atletico Madrid,1800,630,0.12,0.18,0.14,0.20,7.5,22,0.60
ARG_011,Marcos Acuña,Argentina,DEF,Sevilla,1600,540,0.08,0.14,0.09,0.16,8.4,45,0.58
BRA_001,Neymar Jr,Brazil,FWD,Paris Saint-Germain,1800,720,0.55,0.48,0.62,0.55,3.8,120,0.93
BRA_002,Vinicius Jr,Brazil,FWD,Real Madrid,2300,810,0.52,0.38,0.58,0.42,3.5,22,0.88
BRA_003,Richarlison,Brazil,FWD,Tottenham,1900,750,0.45,0.22,0.50,0.25,4.2,40,0.75
BRA_004,Lucas Paquetá,Brazil,MID,West Ham,2100,780,0.22,0.28,0.25,0.32,5.8,44,0.70
BRA_005,Casemiro,Brazil,MID,Manchester United,2200,810,0.08,0.10,0.09,0.12,9.5,65,0.72
BRA_006,Thiago Silva,Brazil,DEF,Chelsea,2100,720,0.04,0.05,0.05,0.06,8.8,108,0.70
BRA_007,Marquinhos,Brazil,DEF,Paris Saint-Germain,2000,750,0.06,0.04,0.07,0.05,9.2,71,0.68
BRA_008,Alisson,Brazil,GK,Liverpool,2700,900,0.00,0.00,0.00,0.00,1.0,62,0.82
BRA_009,Raphinha,Brazil,MID,Barcelona,2000,720,0.35,0.30,0.38,0.33,4.5,30,0.72
BRA_010,Alex Sandro,Brazil,DEF,Juventus,1700,540,0.06,0.12,0.07,0.14,7.8,42,0.60
BRA_011,Antony,Brazil,FWD,Ajax,1900,630,0.30,0.22,0.33,0.24,3.9,18,0.65
FRA_001,Kylian Mbappé,France,FWD,Paris Saint-Germain,2400,900,0.75,0.42,0.82,0.48,4.2,59,0.95
FRA_002,Karim Benzema,France,FWD,Real Madrid,2100,0,0.72,0.35,0.80,0.38,3.5,97,0.90
FRA_003,Antoine Griezmann,France,MID,Atletico Madrid,2200,810,0.32,0.35,0.35,0.38,5.8,111,0.78
FRA_004,Olivier Giroud,France,FWD,AC Milan,1800,630,0.45,0.15,0.48,0.18,3.2,115,0.70
FRA_005,Raphaël Varane,France,DEF,Manchester United,1900,720,0.04,0.03,0.05,0.04,9.5,91,0.72
FRA_006,Dayot Upamecano,France,DEF,Bayern Munich,2100,750,0.04,0.02,0.05,0.03,9.8,23,0.65
FRA_007,Aurélien Tchouaméni,France,MID,Real Madrid,2000,720,0.08,0.10,0.09,0.12,8.5,21,0.68
FRA_008,Adrien Rabiot,France,MID,Juventus,2100,720,0.12,0.15,0.14,0.17,7.2,34,0.65
FRA_009,Hugo Lloris,France,GK,Tottenham,2400,900,0.00,0.00,0.00,0.00,1.1,143,0.78
FRA_010,Théo Hernández,France,DEF,AC Milan,2100,720,0.10,0.18,0.12,0.20,8.2,20,0.65
FRA_011,Ousmane Dembélé,France,FWD,Barcelona,1800,630,0.28,0.28,0.30,0.30,3.5,40,0.68
ENG_001,Harry Kane,England,FWD,Tottenham,2400,900,0.68,0.28,0.75,0.30,3.8,74,0.88
ENG_002,Bukayo Saka,England,FWD,Arsenal,2300,810,0.38,0.35,0.42,0.38,4.5,30,0.78
ENG_003,Phil Foden,England,MID,Manchester City,2200,720,0.35,0.32,0.38,0.35,4.8,26,0.78
ENG_004,Jude Bellingham,England,MID,Borussia Dortmund,2300,810,0.22,0.25,0.24,0.27,7.5,22,0.78
ENG_005,Jordan Henderson,England,MID,Liverpool,2000,720,0.08,0.12,0.09,0.14,7.8,73,0.65
ENG_006,John Stones,England,DEF,Manchester City,2000,750,0.06,0.05,0.07,0.06,9.5,62,0.68
ENG_007,Kyle Walker,England,DEF,Manchester City,2100,810,0.04,0.10,0.05,0.12,8.8,73,0.65
ENG_008,Jordan Pickford,England,GK,Everton,2400,900,0.00,0.00,0.00,0.00,1.2,52,0.72
ENG_009,Raheem Sterling,England,FWD,Chelsea,2000,720,0.35,0.30,0.38,0.33,3.8,77,0.72
ENG_010,Luke Shaw,England,DEF,Manchester United,1800,630,0.06,0.15,0.07,0.17,7.5,23,0.62
ENG_011,Declan Rice,England,MID,West Ham,2200,810,0.08,0.10,0.09,0.12,9.0,37,0.70
GER_001,Kai Havertz,Germany,MID,Chelsea,2100,750,0.28,0.22,0.30,0.24,5.5,35,0.72
GER_002,Serge Gnabry,Germany,FWD,Bayern Munich,1900,680,0.38,0.28,0.42,0.30,4.2,35,0.72
GER_003,Thomas Müller,Germany,MID,Bayern Munich,2100,810,0.18,0.35,0.22,0.38,5.8,121,0.75
GER_004,Jamal Musiala,Germany,MID,Bayern Munich,2000,720,0.28,0.25,0.30,0.27,5.5,22,0.75
GER_005,Joshua Kimmich,Germany,MID,Bayern Munich,2300,900,0.08,0.18,0.09,0.20,9.2,70,0.78
GER_006,Antonio Rüdiger,Germany,DEF,Real Madrid,2100,810,0.05,0.04,0.06,0.05,10.2,57,0.70
GER_007,Manuel Neuer,Germany,GK,Bayern Munich,2700,900,0.00,0.00,0.00,0.00,0.9,116,0.82
GER_008,Niklas Süle,Germany,DEF,Borussia Dortmund,1900,680,0.04,0.03,0.05,0.04,9.8,38,0.65
GER_009,Leon Goretzka,Germany,MID,Bayern Munich,2000,720,0.12,0.15,0.14,0.17,8.2,47,0.68
GER_010,Leroy Sané,Germany,FWD,Bayern Munich,2000,680,0.32,0.25,0.35,0.27,4.0,52,0.70
GER_011,David Raum,Germany,DEF,RB Leipzig,2100,720,0.06,0.20,0.07,0.22,8.5,12,0.62
ESP_001,Gavi,Spain,MID,Barcelona,2300,900,0.15,0.25,0.17,0.28,7.8,22,0.80
ESP_002,Pedri,Spain,MID,Barcelona,2000,750,0.12,0.22,0.14,0.25,7.2,19,0.78
ESP_003,Marco Asensio,Spain,MID,Real Madrid,1800,630,0.25,0.18,0.28,0.20,4.5,43,0.65
ESP_004,Ferran Torres,Spain,FWD,Barcelona,1900,680,0.35,0.25,0.38,0.27,4.2,28,0.68
ESP_005,Álvaro Morata,Spain,FWD,Atletico Madrid,1800,680,0.32,0.18,0.35,0.20,3.8,58,0.65
ESP_006,Jordi Alba,Spain,DEF,Barcelona,1900,720,0.08,0.22,0.09,0.24,8.2,89,0.65
ESP_007,Sergio Busquets,Spain,MID,Barcelona,2000,810,0.03,0.10,0.04,0.12,8.8,143,0.72
ESP_008,Aymeric Laporte,Spain,DEF,Manchester City,2000,720,0.05,0.04,0.06,0.05,10.0,27,0.68
ESP_009,Unai Simón,Spain,GK,Athletic Bilbao,2200,810,0.00,0.00,0.00,0.00,1.0,20,0.68
ESP_010,Rodri,Spain,MID,Manchester City,2300,900,0.06,0.12,0.07,0.14,10.5,28,0.78
ESP_011,Dani Carvajal,Spain,DEF,Real Madrid,2000,720,0.05,0.12,0.06,0.14,8.5,47,0.65
POR_001,Cristiano Ronaldo,Portugal,FWD,Manchester United,2200,900,0.55,0.15,0.60,0.18,3.2,190,0.88
POR_002,Bruno Fernandes,Portugal,MID,Manchester United,2400,900,0.35,0.38,0.38,0.42,5.5,48,0.82
POR_003,Bernardo Silva,Portugal,MID,Manchester City,2300,900,0.22,0.28,0.24,0.30,6.2,70,0.80
POR_004,Joao Felix,Portugal,FWD,Atletico Madrid,1900,680,0.32,0.28,0.35,0.30,4.5,32,0.72
POR_005,Ruben Dias,Portugal,DEF,Manchester City,2100,810,0.05,0.03,0.06,0.04,10.5,44,0.75
POR_006,William Carvalho,Portugal,MID,Real Betis,2000,720,0.05,0.08,0.06,0.09,8.8,80,0.62
POR_007,Rui Patricio,Portugal,GK,AS Roma,2400,900,0.00,0.00,0.00,0.00,1.0,101,0.75
POR_008,Joao Cancelo,Portugal,DEF,Manchester City,2200,810,0.08,0.20,0.09,0.22,8.5,47,0.72
POR_009,Rafael Leao,Portugal,FWD,AC Milan,2200,680,0.38,0.28,0.42,0.30,3.8,22,0.72
POR_010,Danilo,Portugal,DEF,Paris Saint-Germain,1900,680,0.04,0.08,0.05,0.09,8.2,40,0.60
POR_011,Vitinha,Portugal,MID,Paris Saint-Germain,2000,630,0.10,0.15,0.12,0.17,7.2,16,0.62
NED_001,Memphis Depay,Netherlands,FWD,Atletico Madrid,1700,630,0.42,0.25,0.45,0.27,4.2,82,0.75
NED_002,Cody Gakpo,Netherlands,FWD,PSV Eindhoven,2200,780,0.48,0.35,0.52,0.38,4.5,16,0.78
NED_003,Virgil van Dijk,Netherlands,DEF,Liverpool,2300,900,0.06,0.04,0.07,0.05,11.0,52,0.80
NED_004,Denzel Dumfries,Netherlands,DEF,Inter Milan,2100,810,0.10,0.18,0.12,0.20,8.5,40,0.68
NED_005,Frenkie de Jong,Netherlands,MID,Barcelona,2200,810,0.10,0.15,0.12,0.17,7.5,43,0.78
NED_006,Daley Blind,Netherlands,DEF,Ajax,1900,680,0.04,0.08,0.05,0.09,8.8,99,0.62
NED_007,Andries Noppert,Netherlands,GK,Heerenveen,1800,540,0.00,0.00,0.00,0.00,0.8,3,0.62
NED_008,Steven Bergwijn,Netherlands,FWD,Ajax,2000,630,0.30,0.22,0.33,0.24,4.0,27,0.65
NED_009,Teun Koopmeiners,Netherlands,MID,Atalanta,2200,720,0.18,0.20,0.20,0.22,7.8,15,0.68
NED_010,Matthijs de Ligt,Netherlands,DEF,Bayern Munich,1900,680,0.05,0.03,0.06,0.04,10.2,44,0.68
NED_011,Xavi Simons,Netherlands,MID,Paris Saint-Germain,1800,540,0.20,0.22,0.22,0.24,5.5,10,0.62
CRO_001,Luka Modrić,Croatia,MID,Real Madrid,2000,810,0.12,0.22,0.14,0.25,7.5,164,0.88
CRO_002,Ivan Perišić,Croatia,MID,Tottenham,2100,810,0.28,0.28,0.30,0.30,6.5,123,0.75
CRO_003,Ante Budimir,Croatia,FWD,Osasuna,1800,580,0.35,0.12,0.38,0.14,3.5,28,0.62
CRO_004,Mateo Kovačić,Croatia,MID,Chelsea,2000,720,0.08,0.15,0.09,0.17,8.0,80,0.68
CRO_005,Dominik Livaković,Croatia,GK,Dinamo Zagreb,2200,810,0.00,0.00,0.00,0.00,0.9,48,0.68
CRO_006,Dejan Lovren,Croatia,DEF,Zenit,1700,580,0.04,0.03,0.05,0.04,9.2,71,0.60
CRO_007,Josip Šutalo,Croatia,DEF,Dinamo Zagreb,1800,540,0.04,0.02,0.05,0.03,9.5,12,0.58
CRO_008,Marko Livaja,Croatia,FWD,Hajduk Split,1600,580,0.35,0.18,0.38,0.20,3.8,18,0.60
CRO_009,Marcelo Brozović,Croatia,MID,Inter Milan,2100,720,0.06,0.12,0.07,0.14,8.2,58,0.70
CRO_010,Joško Gvardiol,Croatia,DEF,RB Leipzig,2100,720,0.05,0.08,0.06,0.09,10.0,22,0.68
CRO_011,Borna Sosa,Croatia,DEF,Stuttgart,1800,540,0.06,0.14,0.07,0.16,7.5,19,0.60
SEN_001,Sadio Mané,Senegal,FWD,Bayern Munich,0,0,0.52,0.32,0.58,0.35,4.8,91,0.90
SEN_002,Ismaila Sarr,Senegal,FWD,Watford,1800,720,0.38,0.25,0.42,0.27,4.5,46,0.70
SEN_003,Cheikhou Kouyaté,Senegal,MID,Nottingham Forest,1700,630,0.05,0.08,0.06,0.09,8.5,88,0.58
SEN_004,Kalidou Koulibaly,Senegal,DEF,Chelsea,2000,720,0.05,0.03,0.06,0.04,10.8,67,0.72
SEN_005,Édouard Mendy,Senegal,GK,Chelsea,2200,810,0.00,0.00,0.00,0.00,1.0,34,0.72
SEN_006,Idrissa Gueye,Senegal,MID,Everton,1900,720,0.05,0.10,0.06,0.12,9.5,80,0.65
SEN_007,Nampalys Mendy,Senegal,MID,Leicester City,1700,580,0.03,0.06,0.04,0.07,8.2,42,0.55
SEN_008,Boulaye Dia,Senegal,FWD,Salernitana,1800,580,0.38,0.20,0.42,0.22,4.0,18,0.65
SEN_009,Pape Abou Cissé,Senegal,DEF,Olympiacos,1700,580,0.04,0.02,0.05,0.03,9.5,35,0.58
SEN_010,Fodé Baldé,Senegal,FWD,Celta Vigo,1600,540,0.28,0.18,0.30,0.20,3.8,30,0.58
SEN_011,Abdou Diallo,Senegal,DEF,Leipzig,1700,540,0.04,0.05,0.05,0.06,9.0,27,0.58
MAR_001,Hakim Ziyech,Morocco,MID,Chelsea,1800,720,0.28,0.30,0.30,0.33,5.2,57,0.72
MAR_002,Youssef En-Nesyri,Morocco,FWD,Sevilla,2000,750,0.35,0.12,0.38,0.14,4.0,45,0.68
MAR_003,Sofiane Boufal,Morocco,MID,Angers,1600,580,0.25,0.28,0.27,0.30,5.0,49,0.65
MAR_004,Noussair Mazraoui,Morocco,DEF,Bayern Munich,2000,720,0.08,0.15,0.09,0.17,8.2,26,0.68
MAR_005,Romain Saiss,Morocco,DEF,Besiktas,1700,630,0.05,0.04,0.06,0.05,10.2,66,0.65
MAR_006,Sofyan Amrabat,Morocco,MID,Fiorentina,2100,810,0.04,0.08,0.05,0.09,10.5,42,0.70
MAR_007,Yassine Bounou,Morocco,GK,Sevilla,2400,900,0.00,0.00,0.00,0.00,0.8,36,0.75
MAR_008,Selim Amallah,Morocco,MID,Standard Liège,1500,450,0.12,0.14,0.13,0.15,7.0,32,0.58
MAR_009,Achraf Hakimi,Morocco,DEF,Paris Saint-Germain,2300,900,0.12,0.22,0.14,0.24,8.8,57,0.78
MAR_010,Jawad El Yamiq,Morocco,DEF,Real Valladolid,1700,540,0.04,0.02,0.05,0.03,9.8,22,0.58
MAR_011,Ibrahim Diaz,Morocco,FWD,Leganes,1400,360,0.22,0.18,0.24,0.20,4.5,16,0.58
JPN_001,Takumi Minamino,Japan,MID,Monaco,1900,720,0.28,0.22,0.30,0.24,5.5,47,0.68
JPN_002,Daichi Kamada,Japan,MID,Eintracht Frankfurt,2100,750,0.22,0.25,0.24,0.27,6.5,30,0.65
JPN_003,Ritsu Doan,Japan,FWD,Freiburg,2200,810,0.35,0.22,0.38,0.24,4.5,30,0.68
JPN_004,Ao Tanaka,Japan,MID,Fortuna Düsseldorf,1800,630,0.08,0.15,0.09,0.17,8.0,24,0.60
JPN_005,Wataru Endo,Japan,MID,Stuttgart,2100,810,0.06,0.10,0.07,0.12,9.5,54,0.65
JPN_006,Maya Yoshida,Japan,DEF,Schalke,1800,720,0.04,0.03,0.05,0.04,9.5,118,0.62
JPN_007,Shuichi Gonda,Japan,GK,Tokyo,1800,810,0.00,0.00,0.00,0.00,0.8,72,0.62
JPN_008,Hiroki Sakai,Japan,DEF,Urawa Red Diamonds,1600,540,0.05,0.10,0.06,0.12,8.0,75,0.58
JPN_009,Ko Itakura,Japan,DEF,Borussia Mönchengladbach,1900,630,0.05,0.04,0.06,0.05,10.0,24,0.62
JPN_010,Junya Ito,Japan,FWD,Stade de Reims,2000,720,0.28,0.28,0.30,0.30,4.2,36,0.65
JPN_011,Takehiro Tomiyasu,Japan,DEF,Arsenal,1900,630,0.04,0.08,0.05,0.09,8.5,36,0.62
URU_001,Luis Suárez,Uruguay,FWD,Nacional,1200,540,0.42,0.28,0.45,0.30,3.5,136,0.78
URU_002,Darwin Núñez,Uruguay,FWD,Liverpool,2000,720,0.48,0.20,0.52,0.22,4.2,22,0.75
URU_003,Federico Valverde,Uruguay,MID,Real Madrid,2400,900,0.18,0.25,0.20,0.27,7.8,44,0.78
URU_004,Rodrigo Bentancur,Uruguay,MID,Tottenham,2100,750,0.08,0.12,0.09,0.14,8.5,48,0.68
URU_005,Diego Godín,Uruguay,DEF,Independiente,1400,540,0.05,0.03,0.06,0.04,10.5,159,0.65
URU_006,José María Giménez,Uruguay,DEF,Atletico Madrid,1900,720,0.05,0.04,0.06,0.05,10.0,53,0.68
URU_007,Fernando Muslera,Uruguay,GK,Galatasaray,2200,810,0.00,0.00,0.00,0.00,0.9,130,0.68
URU_008,Edinson Cavani,Uruguay,FWD,Valencia,1400,450,0.40,0.15,0.43,0.17,3.5,133,0.70
URU_009,Facundo Pellistri,Uruguay,FWD,Manchester United,1600,540,0.22,0.18,0.24,0.20,4.0,15,0.58
URU_010,Matías Viña,Uruguay,DEF,Salernitana,1700,580,0.06,0.12,0.07,0.14,7.8,28,0.58
URU_011,Nahitan Nandez,Uruguay,MID,Cagliari,1800,630,0.08,0.12,0.09,0.14,8.5,42,0.60
USA_001,Christian Pulisic,USA,FWD,Chelsea,1900,810,0.30,0.22,0.33,0.24,4.5,55,0.72
USA_002,Tyler Adams,USA,MID,Leeds United,2000,810,0.05,0.10,0.06,0.12,9.5,24,0.68
USA_003,Weston McKennie,USA,MID,Juventus,1800,720,0.12,0.12,0.13,0.13,7.5,36,0.65
USA_004,Gio Reyna,USA,MID,Borussia Dortmund,1500,540,0.22,0.25,0.24,0.27,5.5,18,0.65
USA_005,Matt Turner,USA,GK,Arsenal,1800,810,0.00,0.00,0.00,0.00,0.8,34,0.60
USA_006,DeAndre Yedlin,USA,DEF,Inter Miami,1600,580,0.04,0.08,0.05,0.09,7.8,72,0.55
USA_007,Walker Zimmerman,USA,DEF,Nashville SC,1800,720,0.04,0.03,0.05,0.04,9.5,36,0.60
USA_008,Sergiño Dest,USA,DEF,AC Milan,1800,630,0.08,0.15,0.09,0.17,7.5,27,0.62
USA_009,Josh Sargent,USA,FWD,Norwich City,1700,580,0.30,0.15,0.33,0.17,4.0,24,0.60
USA_010,Brenden Aaronson,USA,MID,Leeds United,1800,720,0.18,0.18,0.20,0.20,6.5,24,0.62
USA_011,Yunus Musah,USA,MID,Valencia,2000,720,0.06,0.10,0.07,0.12,7.8,18,0.62
WAL_001,Gareth Bale,Wales,FWD,Los Angeles FC,1600,630,0.32,0.20,0.35,0.22,3.5,111,0.78
WAL_002,Aaron Ramsey,Wales,MID,Nice,1400,450,0.14,0.18,0.16,0.20,6.8,73,0.62
WAL_003,Kieffer Moore,Wales,FWD,Bournemouth,1800,630,0.32,0.10,0.35,0.12,4.5,36,0.62
WAL_004,Daniel James,Wales,FWD,Fulham,1800,720,0.22,0.18,0.24,0.20,4.5,44,0.62
WAL_005,Wayne Hennessey,Wales,GK,Nottingham Forest,2000,810,0.00,0.00,0.00,0.00,0.8,108,0.60
WAL_006,Chris Gunter,Wales,DEF,Wimbledon,1500,450,0.02,0.05,0.03,0.06,8.0,109,0.55
WAL_007,Ben Davies,Wales,DEF,Tottenham,1900,720,0.03,0.06,0.04,0.07,9.0,71,0.62
WAL_008,Connor Roberts,Wales,DEF,Burnley,1700,630,0.05,0.10,0.06,0.12,8.0,42,0.58
WAL_009,Ethan Ampadu,Wales,MID,Spezia,1800,630,0.04,0.08,0.05,0.09,8.5,28,0.60
WAL_010,Joe Allen,Wales,MID,Stoke City,1600,540,0.06,0.12,0.07,0.14,7.5,72,0.58
WAL_011,Joe Rodon,Wales,DEF,Rennes,1800,630,0.03,0.02,0.04,0.03,9.8,27,0.58
ECU_001,Enner Valencia,Ecuador,FWD,Fenerbahce,1900,810,0.45,0.18,0.48,0.20,4.5,78,0.72
ECU_002,Gonzalo Plata,Ecuador,FWD,Sporting CP,1800,680,0.28,0.22,0.30,0.24,4.5,30,0.62
ECU_003,Pervis Estupiñán,Ecuador,DEF,Brighton,2100,810,0.08,0.18,0.09,0.20,8.2,34,0.65
ECU_004,Moisés Caicedo,Ecuador,MID,Brighton,2000,720,0.06,0.10,0.07,0.12,9.5,20,0.65
ECU_005,Hernán Galíndez,Ecuador,GK,Aucas,1800,810,0.00,0.00,0.00,0.00,0.8,22,0.58
ECU_006,Piero Hincapié,Ecuador,DEF,Bayer Leverkusen,2000,720,0.04,0.06,0.05,0.07,9.8,20,0.62
ECU_007,Ángelo Preciado,Ecuador,DEF,Genk,1800,680,0.05,0.10,0.06,0.12,8.0,24,0.58
ECU_008,Jhegson Méndez,Ecuador,MID,Los Angeles FC,1700,580,0.05,0.08,0.06,0.09,8.5,50,0.55
ECU_009,Jeremy Sarmiento,Ecuador,MID,Brighton,1600,540,0.18,0.20,0.20,0.22,5.5,14,0.58
ECU_010,Djorkaeff Reasco,Ecuador,FWD,Tigres,1500,540,0.22,0.15,0.24,0.17,4.0,12,0.55
ECU_011,Robert Arboleda,Ecuador,DEF,Sao Paulo,1700,580,0.04,0.02,0.05,0.03,9.5,34,0.55
QAT_001,Akram Afif,Qatar,FWD,Al Sadd,1800,810,0.40,0.28,0.43,0.30,4.5,42,0.55
QAT_002,Almoez Ali,Qatar,FWD,Al Duhail,1700,810,0.38,0.15,0.40,0.17,4.2,60,0.52
QAT_003,Hassan Al-Haydos,Qatar,FWD,Al Sadd,1600,810,0.25,0.22,0.27,0.24,5.0,171,0.52
QAT_004,Abdulaziz Hatem,Qatar,MID,Al Rayyan,1600,720,0.12,0.18,0.13,0.20,6.5,58,0.50
QAT_005,Saad Al Sheeb,Qatar,GK,Al Sadd,1800,900,0.00,0.00,0.00,0.00,0.8,74,0.52
QAT_006,Bassam Al-Rawi,Qatar,DEF,Al Arabi,1600,720,0.03,0.04,0.04,0.05,9.0,50,0.48
QAT_007,Boualem Khoukhi,Qatar,DEF,Al Sadd,1600,720,0.03,0.02,0.04,0.03,9.5,66,0.48
QAT_008,Pedro Miguel,Qatar,DEF,Al Sadd,1600,720,0.04,0.08,0.05,0.09,8.0,52,0.48
QAT_009,Karim Boudiaf,Qatar,MID,Al Duhail,1600,720,0.05,0.08,0.06,0.09,8.5,88,0.50
QAT_010,Homam Ahmed,Qatar,MID,Al Sadd,1400,630,0.08,0.12,0.09,0.13,7.0,28,0.48
QAT_011,Tarek Salman,Qatar,DEF,Al Sadd,1500,630,0.02,0.03,0.03,0.04,8.5,40,0.46
MEX_001,Hirving Lozano,Mexico,FWD,Napoli,2000,810,0.38,0.28,0.42,0.30,4.5,62,0.75
MEX_002,Raúl Jiménez,Mexico,FWD,Wolves,1700,720,0.38,0.20,0.42,0.22,4.0,104,0.70
MEX_003,Héctor Herrera,Mexico,MID,Houston,1700,720,0.08,0.15,0.09,0.17,8.0,103,0.65
MEX_004,Carlos Acevedo,Mexico,GK,Santos Laguna,1800,720,0.00,0.00,0.00,0.00,0.8,18,0.55
MEX_005,Jesús Gallardo,Mexico,DEF,Monterrey,1800,720,0.06,0.12,0.07,0.14,8.0,58,0.58
MEX_006,Edson Álvarez,Mexico,MID,Ajax,2100,810,0.06,0.10,0.07,0.12,9.5,60,0.68
MEX_007,César Montes,Mexico,DEF,Monterrey,1800,680,0.04,0.02,0.05,0.03,9.8,40,0.58
MEX_008,Andrés Guardado,Mexico,MID,Real Betis,1800,720,0.08,0.15,0.09,0.17,7.5,180,0.65
MEX_009,Alexis Vega,Mexico,FWD,Guadalajara,1700,680,0.25,0.20,0.27,0.22,4.5,32,0.60
MEX_010,Héctor Moreno,Mexico,DEF,Monterrey,1600,630,0.04,0.03,0.05,0.04,9.5,129,0.58
MEX_011,Guillermo Ochoa,Mexico,GK,Salernitana,2200,900,0.00,0.00,0.00,0.00,0.9,132,0.68
POL_001,Robert Lewandowski,Poland,FWD,Barcelona,2400,900,0.72,0.22,0.78,0.24,4.0,138,0.90
POL_002,Piotr Zieliński,Poland,MID,Napoli,2200,810,0.18,0.25,0.20,0.27,6.8,76,0.72
POL_003,Grzegorz Krychowiak,Poland,MID,Al Shabab,1700,720,0.06,0.12,0.07,0.14,8.5,97,0.62
POL_004,Arkadiusz Milik,Poland,FWD,Juventus,1700,630,0.40,0.12,0.43,0.14,3.8,66,0.68
POL_005,Wojciech Szczęsny,Poland,GK,Juventus,2700,900,0.00,0.00,0.00,0.00,1.0,63,0.75
POL_006,Matty Cash,Poland,DEF,Aston Villa,2000,680,0.06,0.12,0.07,0.14,8.5,15,0.60
POL_007,Jan Bednarek,Poland,DEF,Southampton,2000,720,0.04,0.02,0.05,0.03,10.0,52,0.62
POL_008,Jakub Kiwior,Poland,DEF,Spezia,1800,540,0.03,0.04,0.04,0.05,9.5,14,0.58
POL_009,Nicola Zalewski,Poland,MID,AS Roma,1900,680,0.12,0.18,0.13,0.20,6.5,20,0.62
POL_010,Krzysztof Piątek,Poland,FWD,Salernitana,1600,540,0.32,0.10,0.35,0.12,3.5,28,0.60
POL_011,Bartosz Bereszyński,Poland,DEF,Sampdoria,1800,630,0.04,0.08,0.05,0.09,7.8,36,0.55
DEN_001,Christian Eriksen,Denmark,MID,Manchester United,2100,810,0.18,0.32,0.20,0.35,6.0,120,0.82
DEN_002,Andreas Christensen,Denmark,DEF,Barcelona,2000,750,0.05,0.03,0.06,0.04,10.5,47,0.70
DEN_003,Pierre-Emile Højbjerg,Denmark,MID,Tottenham,2200,900,0.08,0.12,0.09,0.14,9.0,68,0.72
DEN_004,Kasper Schmeichel,Denmark,GK,Nice,2100,900,0.00,0.00,0.00,0.00,1.0,89,0.70
DEN_005,Simon Kjær,Denmark,DEF,AC Milan,1900,720,0.04,0.04,0.05,0.05,10.2,135,0.68
DEN_006,Joakim Mæhle,Denmark,DEF,Atalanta,2100,810,0.08,0.18,0.09,0.20,8.2,36,0.65
DEN_007,Rasmus Højlund,Denmark,FWD,Sturm Graz,1700,540,0.32,0.18,0.35,0.20,4.2,14,0.62
DEN_008,Thomas Delaney,Denmark,MID,Sevilla,1800,720,0.06,0.10,0.07,0.12,8.5,63,0.62
DEN_009,Mikkel Damsgaard,Denmark,MID,Brentford,1800,630,0.18,0.22,0.20,0.24,6.0,26,0.65
DEN_010,Yussuf Poulsen,Denmark,FWD,RB Leipzig,1700,580,0.25,0.15,0.27,0.17,4.5,62,0.60
DEN_011,Alexander Bah,Denmark,DEF,Benfica,1700,540,0.05,0.10,0.06,0.12,8.0,14,0.55
BEL_001,Kevin De Bruyne,Belgium,MID,Manchester City,2200,720,0.22,0.45,0.25,0.50,6.5,96,0.92
BEL_002,Eden Hazard,Belgium,FWD,Real Madrid,1400,540,0.28,0.30,0.30,0.33,4.5,126,0.75
BEL_003,Romelu Lukaku,Belgium,FWD,Inter Milan,1600,630,0.48,0.15,0.52,0.17,4.2,102,0.80
BEL_004,Yannick Carrasco,Belgium,MID,Atletico Madrid,2000,720,0.22,0.22,0.24,0.24,5.5,69,0.68
BEL_005,Axel Witsel,Belgium,MID,Atletico Madrid,1900,720,0.06,0.10,0.07,0.12,8.5,130,0.65
BEL_006,Jan Vertonghen,Belgium,DEF,Anderlecht,1800,630,0.04,0.04,0.05,0.05,10.0,145,0.65
BEL_007,Toby Alderweireld,Belgium,DEF,Antwerp,1700,580,0.04,0.03,0.05,0.04,10.5,126,0.65
BEL_008,Thibaut Courtois,Belgium,GK,Real Madrid,2700,900,0.00,0.00,0.00,0.00,0.9,97,0.80
BEL_009,Leandro Trossard,Belgium,FWD,Brighton,2100,720,0.28,0.22,0.30,0.24,4.5,22,0.68
BEL_010,Thomas Meunier,Belgium,DEF,Borussia Dortmund,2000,720,0.06,0.12,0.07,0.14,8.0,62,0.62
BEL_011,Dries Mertens,Belgium,FWD,Galatasaray,1700,540,0.30,0.22,0.33,0.24,4.5,102,0.70
CAM_001,Vincent Aboubakar,Cameroon,FWD,Al Nassr,1800,810,0.45,0.18,0.48,0.20,4.5,88,0.68
CAM_002,Bryan Mbeumo,Cameroon,FWD,Brentford,2000,720,0.30,0.22,0.33,0.24,4.5,20,0.62
CAM_003,Eric Maxim Choupo-Moting,Cameroon,FWD,Bayern Munich,1800,720,0.35,0.15,0.38,0.17,4.0,65,0.65
CAM_004,André-Frank Zambo Anguissa,Cameroon,MID,Napoli,2200,810,0.08,0.12,0.09,0.14,9.5,50,0.68
CAM_005,André Onana,Cameroon,GK,Inter Milan,2700,900,0.00,0.00,0.00,0.00,0.9,32,0.72
CAM_006,Nicholas Nkoulou,Cameroon,DEF,Verona,1700,540,0.04,0.02,0.05,0.03,10.2,62,0.58
CAM_007,Michael Ngadeu,Cameroon,DEF,Gent,1700,580,0.04,0.03,0.05,0.04,9.8,60,0.55
CAM_008,Collins Fai,Cameroon,DEF,Standard Liège,1700,580,0.04,0.08,0.05,0.09,8.0,68,0.55
CAM_009,Pierre Kunde,Cameroon,MID,Olympiacos,1700,580,0.05,0.08,0.06,0.09,8.5,32,0.55
CAM_010,Jean-Charles Castelletto,Cameroon,DEF,Nantes,1800,630,0.04,0.02,0.05,0.03,10.0,28,0.55
CAM_011,Martin Hongla,Cameroon,MID,Verona,1600,540,0.04,0.06,0.05,0.07,8.0,34,0.52
CAN_001,Alphonso Davies,Canada,DEF,Bayern Munich,2300,900,0.10,0.28,0.12,0.30,9.0,45,0.80
CAN_002,Jonathan David,Canada,FWD,Lille,2200,810,0.55,0.20,0.60,0.22,4.5,32,0.78
CAN_003,Cyle Larin,Canada,FWD,Club Brugge,1900,720,0.38,0.15,0.42,0.17,4.2,48,0.68
CAN_004,Tajon Buchanan,Canada,FWD,Club Brugge,2000,720,0.25,0.22,0.27,0.24,4.5,30,0.65
CAN_005,Atiba Hutchinson,Canada,MID,Besiktas,1600,720,0.05,0.10,0.06,0.12,8.5,97,0.60
CAN_006,Scott Kennedy,Canada,DEF,Sainte-Étienne,1500,540,0.02,0.03,0.03,0.04,9.0,26,0.52
CAN_007,Milan Borjan,Canada,GK,Red Star Belgrade,1900,900,0.00,0.00,0.00,0.00,0.8,61,0.58
CAN_008,Steven Vitória,Canada,DEF,Marítimo,1600,540,0.03,0.02,0.04,0.03,9.5,22,0.52
CAN_009,Stephen Eustáquio,Canada,MID,Porto,1900,720,0.08,0.12,0.09,0.14,8.0,32,0.62
CAN_010,Liam Fraser,Canada,MID,Sint-Truiden,1500,450,0.05,0.08,0.06,0.09,7.5,22,0.52
CAN_011,Richie Laryea,Canada,DEF,Nottingham Forest,1700,540,0.05,0.10,0.06,0.12,7.8,26,0.55
```

- [ ] **Step 2: Create `docs/player_data_status.md`**

```markdown
# Player Data — Status

## Current Status

**Engineering validity:** ✅ Complete  
**Data validity:** ❌ Placeholder only

## What Is Placeholder

`data/player_profiles.csv` contains manually estimated stats for WC 2022 squad members.
Values are plausible but not derived from real match-by-match records.
The `base_impact_score` is a composite estimate, not computed from verified metrics.

## Notable Placeholder Entries

- **Karim Benzema (France):** `national_team_minutes_last_12_months = 0`, used to simulate
  his pre-tournament withdrawal. This correctly drives a squad_factor reduction for France.
- **Sadio Mané (Senegal):** `minutes_last_90_days = 0`, simulating his withdrawal.

## What Real Data Would Require

| Field | Source |
|---|---|
| Minutes played | FBref (StatsBomb open data) or Transfermarkt |
| Goals/assists per 90 | FBref player pages |
| xG/xA per 90 | StatsBomb open data or Understat |
| Defensive actions | FBref |
| International caps | Wikipedia / national FA records |
| Availability (per match) | Injury reports (Transfermarkt, BBC Sport) |

## Path to Research-Grade Data

1. Register at FBref.com (free)
2. Write `scripts/fetch_player_stats.py` to pull last-season stats per player
3. For per-match availability: scrape Transfermarkt injury history
4. Validate: compare generated base_impact_score to known rankings

Until this is complete, all player impact results are labelled:
"Engineering validation only — player data not yet research-valid."
```

- [ ] **Step 3: Verify the CSV parses correctly**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/player_profiles.csv', comment='#')
print(f'Rows: {len(df)}, Columns: {len(df.columns)}')
print('Columns:', df.columns.tolist())
teams = df.team.unique()
print(f'Teams: {len(teams)}')
print('France players:')
print(df[df.team=='France'][['player_name','position','base_impact_score']].to_string(index=False))
"
```
Expected: columns all present, France players including Benzema.

- [ ] **Step 4: Commit**

```bash
git add data/player_profiles.csv docs/player_data_status.md
git commit -m "data: player_profiles.csv — placeholder squad stats for 32 WC 2022 teams"
```

---

## Task 2: Match Player Availability Dataset

**Files:**
- Create: `data/match_player_availability.csv`

- [ ] **Step 1: Create `scripts/generate_availability.py`**

This script reads `historical_matches.csv` and `player_profiles.csv` and generates a default availability dataset where all expected starters are fit (form_factor = 1.0, availability_factor = 1.0). Then it applies specific overrides for the notable cases.

```python
"""Generate data/match_player_availability.csv for WC 2022 backtest.

Default: all expected starters fit, form_factor = 1.0.
Overrides: Benzema (France) = out all matches; Mané (Senegal) = out all matches;
           a few form variations to demonstrate the engine.

Run from project root:
    python scripts/generate_availability.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

_ROOT = Path(__file__).parent.parent

# Notable unavailability overrides: {player_id: {status, availability_factor}}
# Applied to ALL matches where the player's team plays.
UNAVAILABLE_PLAYERS = {
    "FRA_002": {"availability_status": "out", "availability_factor": 0.0},     # Benzema
    "SEN_001": {"availability_status": "out", "availability_factor": 0.0},     # Mané
}

# Form boosts: {player_id: form_factor} — applied to all matches
# Demonstrates a player in exceptional form
FORM_OVERRIDES = {
    "ARG_001": 1.15,   # Messi in excellent WC form
    "FRA_001": 1.10,   # Mbappé in excellent form
    "MAR_007": 1.12,   # Bounou excellent GK form
    "JPN_003": 1.08,   # Ritsu Doan good form
}


def main():
    hist = pd.read_csv(_ROOT / "data" / "historical_matches.csv")
    profiles = pd.read_csv(_ROOT / "data" / "player_profiles.csv", comment='#')

    rows = []
    for match_idx, match_row in hist.iterrows():
        match_id = match_idx + 1
        date = str(match_row["date"])
        teams = [str(match_row["team_a"]), str(match_row["team_b"])]

        for team in teams:
            team_players = profiles[profiles["team"] == team].copy()
            # Sort by base_impact_score desc, take top 14 (starting XI + 3 key subs)
            team_players = team_players.sort_values("base_impact_score", ascending=False).head(14)

            for _, player in team_players.iterrows():
                pid = player["player_id"]
                is_starter = player.name in team_players.head(11).index

                # Get override if exists
                override = UNAVAILABLE_PLAYERS.get(pid, {})
                status = override.get("availability_status", "fit")
                avail_factor = override.get("availability_factor", 1.0)
                form = FORM_OVERRIDES.get(pid, 1.0)

                rows.append({
                    "match_id": match_id,
                    "date": date,
                    "team": team,
                    "player_id": pid,
                    "expected_starter": is_starter,
                    "availability_status": status,
                    "availability_factor": avail_factor,
                    "form_factor": form,
                })

    df = pd.DataFrame(rows)
    out = _ROOT / "data" / "match_player_availability.csv"

    # Write with warning header
    with open(out, 'w', encoding='utf-8') as f:
        f.write("# WARNING: PLACEHOLDER DATA — not sourced from real pre-match availability records.\n")
        f.write("# Engineering validation only. See docs/player_data_status.md for sourcing requirements.\n")
        df.to_csv(f, index=False)

    print(f"Generated {len(df):,} rows → {out}")
    print(f"Teams covered per match: both teams × 14 players × {len(hist)} matches")

    # Verify: Benzema should be out for all France matches
    france_rows = df[(df.team == "France") & (df.player_id == "FRA_002")]
    print(f"Benzema rows: {len(france_rows)}, all out: {(france_rows.availability_factor == 0).all()}")

    # Verify: Mané should be out for all Senegal matches
    mane_rows = df[(df.team == "Senegal") & (df.player_id == "SEN_001")]
    print(f"Mané rows: {len(mane_rows)}, all out: {(mane_rows.availability_factor == 0).all()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

```bash
python scripts/generate_availability.py
```

Expected:
```
Generated ~11,200 rows → data/match_player_availability.csv
Benzema rows: X, all out: True
Mané rows: X, all out: True
```

- [ ] **Step 3: Verify the output**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/match_player_availability.csv', comment='#')
print(f'Rows: {len(df)}, Columns: {df.columns.tolist()}')
print(f'Unique matches: {df.match_id.nunique()}')
print(f'Status distribution: {df.availability_status.value_counts().to_dict()}')
# Verify Benzema
benzema = df[df.player_id == 'FRA_002']
print(f'Benzema (FRA_002): {len(benzema)} rows, always out: {(benzema.availability_factor == 0.0).all()}')
# Verify Messi form
messi = df[df.player_id == 'ARG_001']
print(f'Messi (ARG_001): form_factor={messi.form_factor.iloc[0]:.2f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_availability.py data/match_player_availability.csv
git commit -m "data: match_player_availability.csv — placeholder availability for 40 WC 2022 matches"
```

---

## Task 3: Player Loader

**Files:**
- Create: `tests/data/test_player_loader.py`
- Create: `src/data/player_loader.py`

- [ ] **Step 1: Write 6 failing tests**

Create `tests/data/test_player_loader.py`:

```python
import pytest
import pandas as pd
from pathlib import Path
from src.data.player_loader import (
    load_player_profiles, load_match_availability,
    get_team_profiles, get_match_availability,
    PlayerProfile, PlayerAvailability,
)


def _make_profiles_csv(tmp_path):
    f = tmp_path / "profiles.csv"
    pd.DataFrame([
        {"player_id": "FRA_001", "player_name": "Mbappé", "team": "France",
         "position": "FWD", "club": "PSG", "minutes_last_90_days": 2400,
         "national_team_minutes_last_12_months": 900, "goals_per_90": 0.75,
         "assists_per_90": 0.42, "xg_per_90": 0.82, "xa_per_90": 0.48,
         "defensive_actions_per_90": 4.2, "international_caps": 59,
         "base_impact_score": 0.95},
        {"player_id": "FRA_002", "player_name": "Benzema", "team": "France",
         "position": "FWD", "club": "Real Madrid", "minutes_last_90_days": 2100,
         "national_team_minutes_last_12_months": 0, "goals_per_90": 0.72,
         "assists_per_90": 0.35, "xg_per_90": 0.80, "xa_per_90": 0.38,
         "defensive_actions_per_90": 3.5, "international_caps": 97,
         "base_impact_score": 0.90},
        {"player_id": "ENG_001", "player_name": "Kane", "team": "England",
         "position": "FWD", "club": "Tottenham", "minutes_last_90_days": 2400,
         "national_team_minutes_last_12_months": 900, "goals_per_90": 0.68,
         "assists_per_90": 0.28, "xg_per_90": 0.75, "xa_per_90": 0.30,
         "defensive_actions_per_90": 3.8, "international_caps": 74,
         "base_impact_score": 0.88},
    ]).to_csv(f, index=False)
    return f


def _make_avail_csv(tmp_path):
    f = tmp_path / "avail.csv"
    pd.DataFrame([
        {"match_id": 1, "date": "2022-11-20", "team": "France", "player_id": "FRA_001",
         "expected_starter": True, "availability_status": "fit",
         "availability_factor": 1.0, "form_factor": 1.10},
        {"match_id": 1, "date": "2022-11-20", "team": "France", "player_id": "FRA_002",
         "expected_starter": True, "availability_status": "out",
         "availability_factor": 0.0, "form_factor": 1.0},
        {"match_id": 2, "date": "2022-11-21", "team": "England", "player_id": "ENG_001",
         "expected_starter": True, "availability_status": "fit",
         "availability_factor": 1.0, "form_factor": 1.0},
    ]).to_csv(f, index=False)
    return f


def test_load_player_profiles_returns_dict(tmp_path):
    f = _make_profiles_csv(tmp_path)
    result = load_player_profiles(f)
    assert isinstance(result, dict)
    assert "FRA_001" in result
    assert isinstance(result["FRA_001"], PlayerProfile)


def test_load_player_profiles_values_correct(tmp_path):
    f = _make_profiles_csv(tmp_path)
    result = load_player_profiles(f)
    assert result["FRA_001"].player_name == "Mbappé"
    assert abs(result["FRA_001"].base_impact_score - 0.95) < 1e-9


def test_load_match_availability_returns_list(tmp_path):
    f = _make_avail_csv(tmp_path)
    result = load_match_availability(f)
    assert isinstance(result, list)
    assert len(result) == 3
    assert isinstance(result[0], PlayerAvailability)


def test_get_team_profiles_filters_and_sorts(tmp_path):
    f = _make_profiles_csv(tmp_path)
    profiles = load_player_profiles(f)
    france = get_team_profiles(profiles, "France")
    assert len(france) == 2
    # Sorted descending by base_impact_score: FRA_001 (0.95) before FRA_002 (0.90)
    assert france[0].player_id == "FRA_001"


def test_get_match_availability_filters_correctly(tmp_path):
    f = _make_avail_csv(tmp_path)
    avail = load_match_availability(f)
    france_m1 = get_match_availability(avail, match_id=1, team="France")
    assert len(france_m1) == 2
    assert all(a.team == "France" for a in france_m1)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_player_profiles(tmp_path / "missing.csv")
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/data/test_player_loader.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/data/player_loader.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_DEFAULT_PROFILES  = Path(__file__).parent.parent.parent / "data" / "player_profiles.csv"
_DEFAULT_AVAIL     = Path(__file__).parent.parent.parent / "data" / "match_player_availability.csv"


@dataclass
class PlayerProfile:
    player_id: str
    player_name: str
    team: str
    position: str
    club: str
    minutes_last_90_days: float
    national_team_minutes_last_12_months: float
    goals_per_90: float
    assists_per_90: float
    xg_per_90: float
    xa_per_90: float
    defensive_actions_per_90: float
    international_caps: int
    base_impact_score: float


@dataclass
class PlayerAvailability:
    match_id: int
    date: str
    team: str
    player_id: str
    expected_starter: bool
    availability_status: str
    availability_factor: float
    form_factor: float


def load_player_profiles(path: Path | None = None) -> dict[str, PlayerProfile]:
    """Load player profiles CSV. Returns {player_id: PlayerProfile}.

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT_PROFILES
    if not p.exists():
        raise FileNotFoundError(f"player_profiles.csv not found: {p}")
    df = pd.read_csv(p, comment='#')
    return {
        str(row["player_id"]): PlayerProfile(
            player_id=str(row["player_id"]),
            player_name=str(row["player_name"]),
            team=str(row["team"]),
            position=str(row["position"]),
            club=str(row["club"]),
            minutes_last_90_days=float(row["minutes_last_90_days"]),
            national_team_minutes_last_12_months=float(row["national_team_minutes_last_12_months"]),
            goals_per_90=float(row["goals_per_90"]),
            assists_per_90=float(row["assists_per_90"]),
            xg_per_90=float(row["xg_per_90"]),
            xa_per_90=float(row["xa_per_90"]),
            defensive_actions_per_90=float(row["defensive_actions_per_90"]),
            international_caps=int(row["international_caps"]),
            base_impact_score=float(row["base_impact_score"]),
        )
        for _, row in df.iterrows()
    }


def load_match_availability(path: Path | None = None) -> list[PlayerAvailability]:
    """Load match player availability CSV. Returns list of all rows.

    Raises:
        FileNotFoundError: if CSV not found.
    """
    p = path if path is not None else _DEFAULT_AVAIL
    if not p.exists():
        raise FileNotFoundError(f"match_player_availability.csv not found: {p}")
    df = pd.read_csv(p, comment='#')
    return [
        PlayerAvailability(
            match_id=int(row["match_id"]),
            date=str(row["date"]),
            team=str(row["team"]),
            player_id=str(row["player_id"]),
            expected_starter=bool(row["expected_starter"]),
            availability_status=str(row["availability_status"]),
            availability_factor=float(row["availability_factor"]),
            form_factor=float(row["form_factor"]),
        )
        for _, row in df.iterrows()
    ]


def get_team_profiles(
    profiles: dict[str, PlayerProfile],
    team: str,
) -> list[PlayerProfile]:
    """Return all PlayerProfiles for a team, sorted by base_impact_score descending."""
    return sorted(
        [p for p in profiles.values() if p.team == team],
        key=lambda p: -p.base_impact_score,
    )


def get_match_availability(
    availability: list[PlayerAvailability],
    match_id: int,
    team: str,
) -> list[PlayerAvailability]:
    """Return availability rows for one team in one match."""
    return [a for a in availability if a.match_id == match_id and a.team == team]
```

- [ ] **Step 4: Run loader tests**

```bash
python -m pytest tests/data/test_player_loader.py -v
```
Expected: 6 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 136 tests pass (130 + 6).

- [ ] **Step 6: Commit**

```bash
git add src/data/player_loader.py tests/data/test_player_loader.py
git commit -m "feat: player loader — load_player_profiles() and load_match_availability()"
```

---

## Task 4: Player Impact Model

**Files:**
- Create: `tests/models/test_player_impact.py`
- Create: `src/models/player_impact.py`

- [ ] **Step 1: Write 8 failing tests**

Create `tests/models/test_player_impact.py`:

```python
import pytest
from src.models.player_impact import compute_squad_factor, apply_player_impact
from src.data.player_loader import PlayerProfile, PlayerAvailability


def _profile(pid, team, score):
    return PlayerProfile(
        player_id=pid, player_name="Test", team=team, position="MID", club="Club",
        minutes_last_90_days=2000, national_team_minutes_last_12_months=700,
        goals_per_90=0.1, assists_per_90=0.1, xg_per_90=0.1, xa_per_90=0.1,
        defensive_actions_per_90=5.0, international_caps=30,
        base_impact_score=score,
    )


def _avail(pid, team, starter=True, avail_factor=1.0, form_factor=1.0):
    return PlayerAvailability(
        match_id=1, date="2022-11-20", team=team, player_id=pid,
        expected_starter=starter, availability_status="fit",
        availability_factor=avail_factor, form_factor=form_factor,
    )


def _xi(team="A", n=11, score=0.65, avail=1.0, form=1.0):
    profiles = [_profile(f"P{i:02d}", team, score) for i in range(n)]
    avails   = [_avail(f"P{i:02d}", team, starter=True,
                        avail_factor=avail, form_factor=form) for i in range(n)]
    return profiles, avails


def test_returns_float():
    p, a = _xi()
    factor = compute_squad_factor(p, a)
    assert isinstance(factor, float)


def test_all_fit_form_1_returns_1():
    """When all 11 starters are fit with form=1.0, squad_factor should be 1.0."""
    p, a = _xi()
    assert abs(compute_squad_factor(p, a) - 1.0) < 1e-6


def test_key_player_out_reduces_factor():
    """Removing highest-impact player reduces squad_factor below 1.0."""
    profiles = [_profile(f"P{i:02d}", "A", 0.65) for i in range(11)]
    # Make one player the star (0.95) and mark them out
    profiles[0] = _profile("STAR", "A", 0.95)
    avails = [_avail(f"P{i:02d}", "A", avail_factor=1.0) for i in range(11)]
    avails[0] = _avail("STAR", "A", avail_factor=0.0)
    factor = compute_squad_factor(profiles, avails)
    assert factor < 1.0


def test_high_form_increases_factor():
    """High form_factor > 1.0 on starters increases squad_factor above 1.0."""
    p, a = _xi(form=1.3)
    factor = compute_squad_factor(p, a)
    assert factor > 1.0


def test_squad_factor_clamped_to_0_85_1_15():
    """squad_factor is clamped to [0.85, 1.15] regardless of extreme inputs."""
    # All players out (extreme low case)
    p, a = _xi(avail=0.0)
    low = compute_squad_factor(p, a)
    assert low >= 0.85

    # Extreme high form
    p2, a2 = _xi(form=3.0)
    high = compute_squad_factor(p2, a2)
    assert high <= 1.15


def test_no_availability_data_returns_1():
    """No availability rows → squad_factor = 1.0 (safe default)."""
    p, _ = _xi()
    assert compute_squad_factor(p, []) == 1.0


def test_no_expected_starters_returns_1():
    """All expected_starter=False → squad_factor = 1.0 (no starting data)."""
    p, _ = _xi()
    avails = [_avail(f"P{i:02d}", "A", starter=False) for i in range(11)]
    assert compute_squad_factor(p, avails) == 1.0


def test_apply_player_impact_modifies_both_teams():
    """apply_player_impact applies independent factors to each team's xG."""
    p_a, a_a = _xi("A", form=1.1)   # slight boost for team A
    p_b, a_b = _xi("B", avail=0.6)  # reduced team B (all doubtful)

    xg_a_orig, xg_b_orig = 1.5, 1.5
    xg_a_new, xg_b_new = apply_player_impact(
        xg_a_orig, xg_b_orig, p_a, p_b, a_a, a_b
    )
    assert xg_a_new > xg_a_orig    # team A boosted
    assert xg_b_new < xg_b_orig    # team B reduced
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/models/test_player_impact.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/models/player_impact.py`**

```python
"""Player impact model: adjust xG based on squad availability and form.

Formula:
    player_match_impact  = base_impact_score * availability_factor * form_factor
    starting_xi_strength = mean(player_match_impact for expected_starter == True)
    baseline_xi_strength = mean(top 11 base_impact_scores for team)
    squad_factor         = starting_xi_strength / baseline_xi_strength
    squad_factor         = clamp(squad_factor, 0.85, 1.15)
    team_xg             *= squad_factor

Data provenance label: PLACEHOLDER — engineering validation only.
"""

from src.data.player_loader import PlayerProfile, PlayerAvailability

XG_MIN         = 0.2
XG_MAX         = 4.5
SQUAD_FACTOR_MIN = 0.85
SQUAD_FACTOR_MAX = 1.15


def compute_squad_factor(
    profiles: list[PlayerProfile],
    availability: list[PlayerAvailability],
) -> float:
    """Compute the squad quality factor for one team in one match.

    Args:
        profiles: All PlayerProfile objects for this team.
        availability: PlayerAvailability rows for this team in this match.

    Returns:
        squad_factor in [0.85, 1.15]. Returns 1.0 if data is insufficient:
        - no availability rows, OR
        - no expected_starter rows in availability.
    """
    if not availability:
        return 1.0

    # Build lookup: player_id -> availability
    avail_map = {a.player_id: a for a in availability}

    # Starting XI strength: weighted impact of expected starters
    starters = [a for a in availability if a.expected_starter]
    if not starters:
        return 1.0

    profile_map = {p.player_id: p for p in profiles}

    starter_impacts = []
    for a in starters:
        p = profile_map.get(a.player_id)
        if p is None:
            continue
        impact = p.base_impact_score * a.availability_factor * a.form_factor
        starter_impacts.append(impact)

    if not starter_impacts:
        return 1.0

    starting_xi_strength = sum(starter_impacts) / len(starter_impacts)

    # Baseline: top 11 players by base_impact_score (assumes all fit)
    sorted_profiles = sorted(profiles, key=lambda p: -p.base_impact_score)
    top_11 = sorted_profiles[:11]
    if not top_11:
        return 1.0

    baseline_xi_strength = sum(p.base_impact_score for p in top_11) / len(top_11)

    if baseline_xi_strength == 0:
        return 1.0

    squad_factor = starting_xi_strength / baseline_xi_strength
    return float(max(SQUAD_FACTOR_MIN, min(SQUAD_FACTOR_MAX, squad_factor)))


def apply_player_impact(
    xg_a: float,
    xg_b: float,
    profiles_a: list[PlayerProfile],
    profiles_b: list[PlayerProfile],
    availability_a: list[PlayerAvailability],
    availability_b: list[PlayerAvailability],
) -> tuple[float, float]:
    """Apply squad_factor adjustments to both teams' xG.

    Each team's factor is computed independently. Output clamped to [XG_MIN, XG_MAX].
    """
    factor_a = compute_squad_factor(profiles_a, availability_a)
    factor_b = compute_squad_factor(profiles_b, availability_b)

    return (
        float(max(XG_MIN, min(XG_MAX, xg_a * factor_a))),
        float(max(XG_MIN, min(XG_MAX, xg_b * factor_b))),
    )
```

- [ ] **Step 4: Run player impact tests**

```bash
python -m pytest tests/models/test_player_impact.py -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 144 tests pass (136 + 8).

- [ ] **Step 6: Commit**

```bash
git add src/models/player_impact.py tests/models/test_player_impact.py
git commit -m "feat: player impact model — compute_squad_factor() and apply_player_impact()"
```

---

## Task 5: Player Impact Runner

**Files:**
- Create: `tests/backtesting/test_player_impact_runner.py`
- Create: `src/backtesting/player_impact_runner.py`

- [ ] **Step 1: Write 5 failing tests**

Create `tests/backtesting/test_player_impact_runner.py`:

```python
import pytest
import pandas as pd
from pathlib import Path
from src.backtesting.player_impact_runner import run_player_impact_backtest
from src.backtesting.runner import MatchResult
from src.data.player_loader import PlayerProfile, PlayerAvailability


def _make_match_results_csv(tmp_path):
    """Minimal match_results CSV for France vs Australia."""
    f = tmp_path / "mr.csv"
    pd.DataFrame([{
        "match_id": 1, "date": "2022-11-22", "team_a": "France", "team_b": "Australia",
        "team_a_goals": 4, "team_b_goals": 1,
        "team_a_elo_pre": 2102.0, "team_b_elo_pre": 1929.8,
        "team_a_goals_for_last_10": 2.0, "team_a_goals_against_last_10": 0.7,
        "team_b_goals_for_last_10": 1.1, "team_b_goals_against_last_10": 1.4,
        "team_a_points_per_game_last_10": 2.5, "team_b_points_per_game_last_10": 1.5,
        "team_a_matches_available": 10, "team_b_matches_available": 10,
    }]).to_csv(f, index=False)
    return f


def _make_strength_params_csv(tmp_path):
    f = tmp_path / "params.csv"
    pd.DataFrame([
        {"team": "France", "alpha_attack": 3.0, "beta_defense": 0.4, "matches_used": 88, "as_of_date": "2022-11-19"},
        {"team": "Australia", "alpha_attack": 1.0, "beta_defense": 1.0, "matches_used": 20, "as_of_date": "2022-11-19"},
    ]).to_csv(f, index=False)
    return f


def _make_profiles_csv(tmp_path):
    f = tmp_path / "profiles.csv"
    rows = []
    for i in range(11):
        rows.append({
            "player_id": f"FRA_{i:03d}", "player_name": f"Player {i}",
            "team": "France", "position": "MID", "club": "Club",
            "minutes_last_90_days": 2000, "national_team_minutes_last_12_months": 700,
            "goals_per_90": 0.2, "assists_per_90": 0.2, "xg_per_90": 0.2,
            "xa_per_90": 0.2, "defensive_actions_per_90": 5.0,
            "international_caps": 40, "base_impact_score": 0.70,
        })
    for i in range(11):
        rows.append({
            "player_id": f"AUS_{i:03d}", "player_name": f"AUS Player {i}",
            "team": "Australia", "position": "MID", "club": "Club",
            "minutes_last_90_days": 1800, "national_team_minutes_last_12_months": 600,
            "goals_per_90": 0.1, "assists_per_90": 0.1, "xg_per_90": 0.1,
            "xa_per_90": 0.1, "defensive_actions_per_90": 4.0,
            "international_caps": 20, "base_impact_score": 0.55,
        })
    pd.DataFrame(rows).to_csv(f, index=False)
    return f


def _make_avail_csv(tmp_path, france_out_player=False):
    f = tmp_path / "avail.csv"
    rows = []
    for i in range(11):
        avail = 0.0 if (france_out_player and i == 0) else 1.0
        status = "out" if (france_out_player and i == 0) else "fit"
        rows.append({
            "match_id": 1, "date": "2022-11-22", "team": "France",
            "player_id": f"FRA_{i:03d}", "expected_starter": True,
            "availability_status": status, "availability_factor": avail, "form_factor": 1.0,
        })
    for i in range(11):
        rows.append({
            "match_id": 1, "date": "2022-11-22", "team": "Australia",
            "player_id": f"AUS_{i:03d}", "expected_starter": True,
            "availability_status": "fit", "availability_factor": 1.0, "form_factor": 1.0,
        })
    pd.DataFrame(rows).to_csv(f, index=False)
    return f


def test_returns_list_of_match_result(tmp_path):
    mr = _make_match_results_csv(tmp_path)
    sp = _make_strength_params_csv(tmp_path)
    pr = _make_profiles_csv(tmp_path)
    av = _make_avail_csv(tmp_path)
    results = run_player_impact_backtest(mr, sp, pr, av)
    assert len(results) == 1
    assert isinstance(results[0], MatchResult)


def test_works_with_poisson(tmp_path):
    mr = _make_match_results_csv(tmp_path)
    sp = _make_strength_params_csv(tmp_path)
    pr = _make_profiles_csv(tmp_path)
    av = _make_avail_csv(tmp_path)
    results = run_player_impact_backtest(mr, sp, pr, av, model_type="poisson")
    assert results[0].win_a_prob > 0


def test_works_with_dixon_coles(tmp_path):
    mr = _make_match_results_csv(tmp_path)
    sp = _make_strength_params_csv(tmp_path)
    pr = _make_profiles_csv(tmp_path)
    av = _make_avail_csv(tmp_path)
    results = run_player_impact_backtest(mr, sp, pr, av, model_type="dixon_coles", rho=-0.30)
    assert results[0].win_a_prob > 0


def test_all_fit_equals_mle_result(tmp_path):
    """When all players fit with form=1.0, result should match MLE without player impact."""
    from src.backtesting.strength_runner import run_strength_backtest
    mr = _make_match_results_csv(tmp_path)
    sp = _make_strength_params_csv(tmp_path)
    pr = _make_profiles_csv(tmp_path)
    av = _make_avail_csv(tmp_path, france_out_player=False)

    result_impact = run_player_impact_backtest(mr, sp, pr, av, model_type="poisson")[0]
    result_mle    = run_strength_backtest(mr, sp, model_type="poisson")[0]
    assert abs(result_impact.win_a_prob - result_mle.win_a_prob) < 1e-6


def test_player_out_changes_probabilities(tmp_path):
    """When a France starter is out, France win probability should decrease."""
    mr = _make_match_results_csv(tmp_path)
    sp = _make_strength_params_csv(tmp_path)
    pr = _make_profiles_csv(tmp_path)

    av_full = _make_avail_csv(tmp_path, france_out_player=False)
    av_missing = _make_avail_csv(tmp_path, france_out_player=True)

    result_full    = run_player_impact_backtest(mr, sp, pr, av_full, model_type="poisson")[0]
    result_missing = run_player_impact_backtest(mr, sp, pr, av_missing, model_type="poisson")[0]
    assert result_missing.win_a_prob < result_full.win_a_prob
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/backtesting/test_player_impact_runner.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create `src/backtesting/player_impact_runner.py`**

```python
"""Run backtest using MLE strength + player impact adjustment.

Data provenance:
    - match_results.csv: real historical pre-match stats (Sprint 1)
    - team_strength_params.csv: MLE-fitted parameters (Sprint 1/2)
    - player_profiles.csv: PLACEHOLDER — not research-valid
    - match_player_availability.csv: PLACEHOLDER — not research-valid

Results labelled: "Engineering validation only — player data not yet research-valid."
"""

from pathlib import Path
import pandas as pd

from src.data.match_resolver import resolve_all_matches
from src.data.strength_loader import load_strength_params
from src.data.player_loader import load_player_profiles, load_match_availability, get_team_profiles, get_match_availability
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.models.player_impact import apply_player_impact
from src.models.poisson import predict
from src.models.dixon_coles import predict_dixon_coles
from src.backtesting.runner import MatchResult

_ROOT = Path(__file__).parent.parent.parent


def run_player_impact_backtest(
    match_results_path: Path | None = None,
    strength_params_path: Path | None = None,
    player_profiles_path: Path | None = None,
    availability_path: Path | None = None,
    model_type: str = "dixon_coles",
    rho: float = -0.30,
) -> list[MatchResult]:
    """Run backtest with MLE strength + player impact xG modifier.

    Falls back to squad_factor = 1.0 (no player adjustment) for any match
    where player availability data is missing for a team.

    Args:
        match_results_path: Path to match_results.csv (real data).
        strength_params_path: Path to team_strength_params.csv (normalized MLE).
        player_profiles_path: Path to player_profiles.csv (placeholder).
        availability_path: Path to match_player_availability.csv (placeholder).
        model_type: "poisson" or "dixon_coles".
        rho: Dixon-Coles rho parameter.

    Returns:
        list[MatchResult] — compatible with compute_metrics() and all downstream analysis.
    """
    if model_type not in ("poisson", "dixon_coles"):
        raise ValueError(f"model_type must be 'poisson' or 'dixon_coles', got '{model_type}'")

    mr_path = match_results_path or (_ROOT / "data" / "match_results.csv")
    sp_path = strength_params_path or (_ROOT / "data" / "team_strength_params.csv")
    pp_path = player_profiles_path or (_ROOT / "data" / "player_profiles.csv")
    av_path = availability_path or (_ROOT / "data" / "match_player_availability.csv")

    match_results_df = pd.read_csv(mr_path)
    strength         = load_strength_params(sp_path)
    profiles         = load_player_profiles(pp_path)
    availability     = load_match_availability(av_path)

    hist = pd.read_csv(_ROOT / "data" / "historical_matches.csv")
    resolved, _ = resolve_all_matches(hist, match_results_df)

    results = []
    for match_idx, hist_row in hist.iterrows():
        match_id = match_idx + 1
        team_a = str(hist_row["team_a"])
        team_b = str(hist_row["team_b"])

        # Find resolved stats for this match
        resolved_stats = next(
            (r for r in resolved if r.team_a == team_a and r.team_b == team_b),
            None,
        )
        if resolved_stats is None:
            continue

        if team_a not in strength or team_b not in strength:
            continue

        # Step 1: MLE strength-adjusted xG
        xg_a, xg_b = calculate_strength_adjusted_xg(
            resolved_stats.team_a_elo_pre,
            resolved_stats.team_b_elo_pre,
            strength[team_a],
            strength[team_b],
            resolved_stats.team_a_points_per_game_last_10,
            resolved_stats.team_b_points_per_game_last_10,
        )

        # Step 2: Player impact adjustment
        profiles_a  = get_team_profiles(profiles, team_a)
        profiles_b  = get_team_profiles(profiles, team_b)
        avail_a     = get_match_availability(availability, match_id, team_a)
        avail_b     = get_match_availability(availability, match_id, team_b)

        xg_a, xg_b = apply_player_impact(xg_a, xg_b, profiles_a, profiles_b, avail_a, avail_b)

        # Step 3: Predict
        if model_type == "dixon_coles":
            prediction = predict_dixon_coles(team_a, team_b, xg_a, xg_b, rho=rho)
        else:
            prediction = predict(team_a, team_b, xg_a, xg_b)

        ga = int(hist_row["team_a_goals"])
        gb = int(hist_row["team_b_goals"])

        if ga > gb: actual = "team_a_win"
        elif ga == gb: actual = "draw"
        else: actual = "team_b_win"

        probs = {"team_a_win": prediction.win_a, "draw": prediction.draw, "team_b_win": prediction.win_b}
        top5 = [(g_a, g_b) for g_a, g_b, _ in prediction.top_scorelines]

        results.append(MatchResult(
            date=resolved_stats.date, team_a=team_a, team_b=team_b,
            actual_goals_a=ga, actual_goals_b=gb, actual_outcome=actual,
            win_a_prob=prediction.win_a, draw_prob=prediction.draw, win_b_prob=prediction.win_b,
            predicted_outcome=max(probs, key=probs.get),
            top_scorelines=prediction.top_scorelines,
            exact_score_hit=len(top5) > 0 and top5[0] == (ga, gb),
            in_top_3=(ga, gb) in top5[:3],
            in_top_5=(ga, gb) in top5,
            prob_of_actual_result=probs[actual],
        ))

    return results
```

- [ ] **Step 4: Run runner tests**

```bash
python -m pytest tests/backtesting/test_player_impact_runner.py -v
```
Expected: 5 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 149 tests pass (144 + 5).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/player_impact_runner.py tests/backtesting/test_player_impact_runner.py
git commit -m "feat: player impact runner — MLE + squad availability xG adjustment"
```

---

## Task 6: Sprint 3 Report

**Files:**
- Create: `scripts/run_sprint3_report.py`

- [ ] **Step 1: Create `scripts/run_sprint3_report.py`**

```python
"""Sprint 3 validation: compare MLE+DC vs MLE+DC+PlayerImpact on 40 WC 2022 matches.

Run from project root:
    python scripts/run_sprint3_report.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.strength_runner import run_strength_backtest
from src.backtesting.player_impact_runner import run_player_impact_backtest
from src.backtesting.metrics import compute_metrics
from src.data.player_loader import load_player_profiles, load_match_availability, get_team_profiles, get_match_availability
from src.models.strength_adjusted_xg import calculate_strength_adjusted_xg
from src.data.strength_loader import load_strength_params
from src.data.match_resolver import resolve_all_matches

_ROOT = Path(__file__).parent.parent
_RHO = -0.30  # Best rho from Sprint 2


def print_row(label, m, n):
    print(f"  {label:<45s}  {n:>4}  {m.accuracy_1x2:>6.1%}  {m.brier_score:>7.4f}  "
          f"{m.exact_score_accuracy:>6.1%}  {m.top_3_hit_rate:>6.1%}  "
          f"{m.top_5_hit_rate:>6.1%}  {m.avg_prob_actual_result:>6.1%}")


def show_squad_factor_examples():
    """Print per-match squad factor for France (Benzema missing) and Senegal (Mané missing)."""
    hist     = pd.read_csv(_ROOT / "data" / "historical_matches.csv")
    mr       = pd.read_csv(_ROOT / "data" / "match_results.csv")
    strength = load_strength_params()
    profiles = load_player_profiles()
    avail    = load_match_availability()
    resolved, _ = resolve_all_matches(hist, mr)

    print()
    print("Squad Factor Examples (Placeholder Data):")
    print(f"  {'Match':<35s}  {'Team':<12s}  {'squad_factor':>12}")
    print("-" * 65)

    from src.models.player_impact import compute_squad_factor

    interesting_teams = {
        "France": "Benzema OUT (FRA_002)",
        "Senegal": "Mané OUT (SEN_001)",
        "Argentina": "Messi form=1.15",
    }

    for match_idx, hist_row in hist.iterrows():
        match_id = match_idx + 1
        match_label = f"{hist_row['team_a']} vs {hist_row['team_b']}"

        for team, note in interesting_teams.items():
            if hist_row["team_a"] != team and hist_row["team_b"] != team:
                continue

            p = get_team_profiles(profiles, team)
            a = get_match_availability(avail, match_id, team)
            if not a:
                continue
            factor = compute_squad_factor(p, a)
            if abs(factor - 1.0) > 0.001:  # Only show when factor differs
                print(f"  {match_label:<35s}  {team:<12s}  {factor:>12.4f}  ({note})")

        if match_id >= 10:
            break  # Show first 10 matches only


def main():
    print("⚠️  WARNING: Player data is PLACEHOLDER — engineering validation only.")
    print()
    print("Loading data and running backtests...")

    # Model 3: MLE + DC (Sprint 2 best)
    m3_results = run_strength_backtest(model_type="dixon_coles", rho=_RHO)
    m3 = compute_metrics(m3_results)

    # Model 5: MLE + DC + Player Impact
    m5_results = run_player_impact_backtest(model_type="dixon_coles", rho=_RHO)
    m5 = compute_metrics(m5_results)

    print()
    print("=" * 105)
    print("  SPRINT 3 BACKTEST — WC 2022 (⚠️ Player data: PLACEHOLDER)")
    print("=" * 105)
    print(f"  {'Model':<45s}  {'N':>4}  {'1X2':>6}  {'Brier':>7}  {'Exact':>6}  "
          f"{'Top3':>6}  {'Top5':>6}  {'AvgP':>6}")
    print("-" * 105)
    print_row("3. MLE + Dixon-Coles rho=-0.30 (Sprint 2)", m3, m3.total_matches)
    print_row("5. MLE + DC + Player Impact (PLACEHOLDER)", m5, m5.total_matches)
    print("=" * 105)
    print()

    delta = m3.brier_score - m5.brier_score
    print(f"Brier delta (M3 → M5): {delta:+.4f}  "
          f"({'player impact helps' if delta > 0 else 'player impact hurts or neutral'})")
    print()
    print("Note: Delta is NOT meaningful with placeholder player data.")
    print("      The pipeline is validated structurally.")

    show_squad_factor_examples()

    print()
    print("Match examples where player impact changed xG (Benzema, Mané):")
    _show_probability_changes(m3_results, m5_results)


def _show_probability_changes(m3_results, m5_results):
    """Print matches where win_a_prob changed between model 3 and model 5."""
    by_match = {(r.team_a, r.team_b): r for r in m3_results}
    print(f"  {'Match':<35s}  {'MLE+DC win_a':>12}  {'+ Impact win_a':>14}  {'delta':>8}")
    print("-" * 75)
    shown = 0
    for r5 in m5_results:
        r3 = by_match.get((r5.team_a, r5.team_b))
        if r3 is None:
            continue
        delta = r5.win_a_prob - r3.win_a_prob
        if abs(delta) > 0.002:  # Only show meaningful changes
            label = f"{r5.team_a} vs {r5.team_b}"
            print(f"  {label:<35s}  {r3.win_a_prob:>12.1%}  {r5.win_a_prob:>14.1%}  {delta:>+8.2%}")
            shown += 1
    if shown == 0:
        print("  (No matches with delta > 0.2% — all starters fit or no data)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report**

```bash
python scripts/run_sprint3_report.py
```

- [ ] **Step 3: Run full suite — no regressions**

```bash
python -m pytest -v 2>&1 | tail -5
```
Expected: 149 tests pass.

- [ ] **Step 4: Final commit**

```bash
git add scripts/run_sprint3_report.py
git commit -m "feat: Sprint 3 complete — player impact pipeline with comparison report"
```

---

## Task 7: Final Verification

- [ ] **Step 1: All success criteria**

```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.data.player_loader import load_player_profiles, load_match_availability
from src.models.player_impact import compute_squad_factor, apply_player_impact

p = load_player_profiles()
a = load_match_availability()

# Check: Benzema out
fra_m1 = [x for x in a if x.team=='France' and x.match_id==1]
benzema = next((x for x in fra_m1 if x.player_id=='FRA_002'), None)
print(f'Benzema (match 1): availability={benzema.availability_factor if benzema else \"NOT FOUND\"} (expect 0.0)')

# Check: squad_factor != 1.0 for France (Benzema missing)
fra_profiles = [v for v in p.values() if v.team=='France']
sf = compute_squad_factor(fra_profiles, fra_m1)
print(f'France squad_factor match 1: {sf:.4f} (expect < 1.0)')

# Check: squad_factor = 1.0 when all fit
from src.data.player_loader import PlayerProfile, PlayerAvailability
fake_avail = [PlayerAvailability(1,'2022-11-22','T',f'P{i}',True,'fit',1.0,1.0) for i in range(11)]
from tests.models.test_player_impact import _profile
fake_prof = [_profile(f'P{i}','T',0.65) for i in range(11)]
sf2 = compute_squad_factor(fake_prof, fake_avail)
print(f'All-fit squad_factor: {sf2:.6f} (expect 1.0)')
"
```

- [ ] **Step 2: Final test run**

```bash
python -m pytest -v
```
Expected: 149 tests pass.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: Phase 2 Sprint 3 complete — player impact engine, structural validation"
```
