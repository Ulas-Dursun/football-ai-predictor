from pathlib import Path
import socket
import traceback  # Hata ayıklama için eklendi

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    MarketProbability,
    LeagueResponse,
    TeamsResponse,
)
from src.domain.match import MatchInput
from src.application.prediction_service import PredictionService

app = FastAPI(title="Football Betting ML API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent / "templates")
)

service = PredictionService()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, league: str | None = None):
    try:
        leagues = service.get_leagues()
        if not leagues:
            selected_league = None
        else:
            selected_league = league if league in leagues else leagues[0]

        teams = service.get_teams_for_league(selected_league) if selected_league else []

        return templates.TemplateResponse("index.html", {
            "request": request,
            "leagues": leagues,
            "selected_league": selected_league,
            "teams": teams,
            "selected_home": None,
            "selected_away": None,
            "predictions": None,
            "suggestions": None,
            "home_team": None,
            "away_team": None,
            "error": None,
            "max_prob": None,
        })
    except Exception as e:
        print(f"HOME ERROR: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request, "leagues": [], "error": f"Sistem Hatası: {str(e)}"
        })


@app.post("/", response_class=HTMLResponse)
def predict_form(
        request: Request,
        league: str = Form(...),
        home_team: str = Form(...),
        away_team: str = Form(...),
):
    leagues = service.get_leagues()
    teams = service.get_teams_for_league(league)

    if home_team == away_team:
        return templates.TemplateResponse("index.html", {
            "request": request, "leagues": leagues, "selected_league": league,
            "teams": teams, "selected_home": home_team, "selected_away": away_team,
            "predictions": None, "error": "Ev sahibi ve deplasman aynı olamaz!",
        })

    try:
        match = MatchInput(league=league, home_team=home_team, away_team=away_team)

        # Servisten sonuçları al
        result_data = service.predict_for_match(match)

        # Verileri değişkenlere çıkar
        preds = result_data.get("predictions", [])
        home_form = result_data.get("home_form", [])
        away_form = result_data.get("away_form", [])
        home_history = result_data.get("home_history", [])  # Listeyi alıyoruz
        away_history = result_data.get("away_history", [])  # Listeyi alıyoruz

        suggestions = service.generate_suggestions(preds)
        max_prob = max((p.probability for p in preds), default=None)
        error = None

    except Exception as e:
        import traceback
        traceback.print_exc()
        preds, suggestions, home_form, away_form = [], [], [], []
        home_history, away_history = [], []
        error = f"Analiz Hatası: {str(e)}"
        max_prob = None

    return templates.TemplateResponse("index.html", {
        "request": request,
        "leagues": leagues,
        "selected_league": league,
        "teams": teams,
        "selected_home": home_team,
        "selected_away": away_team,
        "predictions": preds,
        "suggestions": suggestions,
        "home_form": home_form,
        "away_form": away_form,
        "home_history": home_history,  # HTML'e giden kritik veri
        "away_history": away_history,  # HTML'e giden kritik veri
        "error": error,
        "max_prob": max_prob,
    })

# JSON API Routes (Aynı kalabilir)
@app.get("/leagues", response_model=LeagueResponse)
def list_leagues():
    return LeagueResponse(leagues=service.get_leagues())


@app.get("/teams", response_model=TeamsResponse)
def list_teams(league: str):
    teams = service.get_teams_for_league(league)
    return TeamsResponse(league=league, teams=teams)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.home_team == request.away_team:
        raise HTTPException(status_code=400, detail="Teams must be different")
    try:
        match = MatchInput(league=request.league, home_team=request.home_team, away_team=request.away_team)
        raw_preds = service.predict_for_match(match)
        markets = [
            MarketProbability(
                market=p.market, probability=p.probability,
                probability_percent=p.probability_percent,
                fair_odds=p.fair_odds, risk_color=p.risk_color
            ) for p in raw_preds
        ]
        return PredictResponse(
            league=request.league, home_team=request.home_team,
            away_team=request.away_team, markets=markets
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)