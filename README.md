# World Cup Predictor

A local Python MVP that predicts World Cup match outcomes using a Poisson distribution model.

## How It Works

Given two teams and their expected goals (xG), the model builds a score probability matrix using independent Poisson distributions. From that matrix it derives win/draw/loss probabilities and the five most likely exact scorelines.

## Setup

1. **Clone the repo and enter the directory:**
   ```bash
   cd worldcup-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run src/app/app.py
   ```

4. Open the URL shown in the terminal (default: http://localhost:8501).

## Running Tests

```bash
pytest -v
```

## Project Structure

```
worldcup-predictor/
├── data/
│   └── teams.csv          # National teams list — replace to add/remove teams
├── src/
│   ├── data/
│   │   └── loader.py      # Data seam — swap this file to load from API or DB
│   ├── models/
│   │   └── poisson.py     # Poisson prediction model
│   ├── app/
│   │   └── app.py         # Streamlit UI
│   └── backtesting/       # Reserved for v2
├── requirements.txt
└── README.md
```

## Expanding to Future Versions

- **New teams or ratings:** Replace `src/data/loader.py` — it is the only file that knows where data comes from.
- **Improved model:** Extend `PredictionResult` in `src/models/poisson.py` with ELO adjustments, form factors, or player ratings.
- **Backtesting:** Add historical match replay logic in `src/backtesting/`.
