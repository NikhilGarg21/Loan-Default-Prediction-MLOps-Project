from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import LoanApplicantData, LoanDefaultClassifier
from src.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """
    DataForm class to handle and process incoming form data.
    """

    def __init__(self, request: Request):
        self.request: Request = request
        self.Age: Optional[int] = None
        self.Income: Optional[float] = None
        self.LoanAmount: Optional[float] = None
        self.CreditScore: Optional[int] = None
        self.MonthsEmployed: Optional[int] = None
        self.NumCreditLines: Optional[int] = None
        self.InterestRate: Optional[float] = None
        self.LoanTerm: Optional[int] = None
        self.DTIRatio: Optional[float] = None
        self.Education: Optional[int] = None
        self.EmploymentType: Optional[str] = None
        self.MaritalStatus: Optional[str] = None
        self.HasMortgage: Optional[int] = None
        self.HasDependents: Optional[int] = None
        self.LoanPurpose: Optional[str] = None
        self.HasCoSigner: Optional[int] = None

    async def get_loan_data(self):
        """
        Retrieve form data and convert it to
        appropriate Python data types.
        """
        form = await self.request.form()
        self.Age = int(form.get("Age"))
        self.Income = float(form.get("Income"))
        self.LoanAmount = float(form.get("LoanAmount"))
        self.CreditScore = int(form.get("CreditScore"))
        self.MonthsEmployed = int(form.get("MonthsEmployed"))
        self.NumCreditLines = int(form.get("NumCreditLines"))
        self.InterestRate = float(form.get("InterestRate"))
        self.LoanTerm = int(form.get("LoanTerm"))
        self.DTIRatio = float(form.get("DTIRatio"))
        self.Education = int(form.get("Education"))
        self.EmploymentType = form.get("EmploymentType")
        self.MaritalStatus = form.get("MaritalStatus")
        self.HasMortgage = int(form.get("HasMortgage"))
        self.HasDependents = int(form.get("HasDependents"))
        self.LoanPurpose = form.get("LoanPurpose")
        self.HasCoSigner = int(form.get("HasCoSigner"))


@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Render the main loan prediction page.
    """
    return templates.TemplateResponse(
        request=request, name="index.html", context={"context": None}
    )

@app.post("/")
async def predictRouteClient(request: Request):
    """
    Receive form data and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_loan_data()
        loan_data = LoanApplicantData(
            Age=form.Age,
            Income=form.Income,
            LoanAmount=form.LoanAmount,
            CreditScore=form.CreditScore,
            MonthsEmployed=form.MonthsEmployed,
            NumCreditLines=form.NumCreditLines,
            InterestRate=form.InterestRate,
            LoanTerm=form.LoanTerm,
            DTIRatio=form.DTIRatio,
            Education=form.Education,
            EmploymentType=form.EmploymentType,
            MaritalStatus=form.MaritalStatus,
            HasMortgage=form.HasMortgage,
            HasDependents=form.HasDependents,
            LoanPurpose=form.LoanPurpose,
            HasCoSigner=form.HasCoSigner,
        )

        loan_df = loan_data.get_loan_input_data_frame()
        model_predictor = LoanDefaultClassifier()
        value = model_predictor.predict(dataframe=loan_df)[0]

        status = "Will Default" if value == 1 else "Will Repay"

        return templates.TemplateResponse(
            request=request, name="result.html", context={"status": status}
        )

    except Exception as e:
        return Response(content=f"Error Occurred! {str(e)}", status_code=500)

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
