import os
import json
import uuid

import numpy as np
import tensorflow as tf

from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify
)

from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from dotenv import load_dotenv

from prediction import predict_image
from gradcam import make_gradcam
from pdf_report import generate_pdf

from ai_assistant import (
    explain_report,
    chat_with_ai,
    chat_about_report
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)


# ============================================================
# CONFIGURATION
# ============================================================

app.secret_key = os.getenv(
    "SECRET_KEY",
    "development-secret-key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///"
    + os.path.join(
        BASE_DIR,
        "brainscan.db"
    )
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

app.config["RESULT_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

app.config["MODEL_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "models"
)

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


# ============================================================
# DATABASE
# ============================================================

db = SQLAlchemy(app)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)

os.makedirs(
    app.config["RESULT_FOLDER"],
    exist_ok=True
)

os.makedirs(
    app.config["MODEL_FOLDER"],
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    app.config["MODEL_FOLDER"],
    "brain_tumor_mobilenetv2.keras"
)

model = None


if os.path.exists(MODEL_PATH):

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print(
            "MobileNetV2 model loaded successfully."
        )

    except Exception as e:

        print(
            "ERROR loading model:",
            e
        )

else:

    print(
        "WARNING: MobileNetV2 model not found:"
    )

    print(MODEL_PATH)


# ============================================================
# DATABASE MODELS
# ============================================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        default="patient",
        nullable=False
    )

    reports = db.relationship(
        "Report",
        backref="user",
        lazy=True
    )


class Report(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    image = db.Column(
        db.String(255),
        nullable=False
    )

    prediction = db.Column(
        db.String(100),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    probabilities = db.Column(
        db.Text,
        nullable=True
    )

    gradcam = db.Column(
        db.String(255),
        nullable=True
    )

    review_status = db.Column(
        db.String(50),
        default="Pending"
    )

    doctor_comment = db.Column(
        db.Text,
        nullable=True
    )


# ============================================================
# CREATE DATABASE
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


def validate_image(file):

    try:

        image = Image.open(file)

        image.verify()

        file.seek(0)

        return True

    except Exception:

        return False


def login_required():

    return "user_id" in session


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", "patient")

        # Check required fields
        if not name or not email or not password:
            return render_template(
                "register.html",
                error="All fields are required."
            )

        # Check password
        if password != confirm_password:
            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        # Minimum password length
        if len(password) < 8:
            return render_template(
                "register.html",
                error="Password must contain at least 8 characters."
            )

        # Only allow valid roles
        if role not in ["patient", "doctor"]:
            role = "patient"

        # Check existing account
        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:
            return render_template(
                "register.html",
                error="Email already registered."
            )

        # Hash password
        hashed_password = generate_password_hash(password)

        # Create user
        user = User(
            name=name,
            email=email,
            password=hashed_password,
            role=role
        )

        db.session.add(user)
        db.session.commit()

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )
# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if user:

            try:

                password_correct = (
                    check_password_hash(
                        user.password,
                        password
                    )
                )

            except ValueError:

                password_correct = False

            if password_correct:

                session["user_id"] = user.id

                session["role"] = user.role

                session["name"] = user.name

                if user.role == "doctor":

                    return redirect(
                        url_for(
                            "doctor_dashboard"
                        )
                    )

                if user.role == "admin":

                    return redirect(
                        url_for(
                            "admin_dashboard"
                        )
                    )

                return redirect(
                    url_for(
                        "patient_dashboard"
                    )
                )

        return render_template(
            "login.html",
            error="Invalid email or password."
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# PATIENT DASHBOARD
# ============================================================

@app.route(
    "/patient-dashboard"
)
def patient_dashboard():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if session.get("role") != "patient":

        return "Access denied", 403

    user = User.query.get(
        session["user_id"]
    )

    reports = Report.query.filter_by(
        user_id=user.id
    ).order_by(
        Report.id.desc()
    ).all()

    return render_template(
        "patient_dashboard.html",
        user=user,
        reports=reports
    )


# ============================================================
# MRI PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if session.get("role") != "patient":

        return "Access denied", 403

    if model is None:

        return render_template(
            "error.html",
            message=(
                "AI model is not available. "
                "Please check the model file."
            )
        ), 500

    if "image" not in request.files:

        return (
            "Please upload an MRI image.",
            400
        )

    file = request.files["image"]

    if file.filename == "":

        return (
            "No image selected.",
            400
        )

    if not allowed_file(
        file.filename
    ):

        return (
            "Only JPG, JPEG and PNG "
            "images are allowed.",
            400
        )

    if not validate_image(file):

        return (
            "Invalid or corrupted image.",
            400
        )

    original_name = secure_filename(
        file.filename
    )

    filename = (
        str(uuid.uuid4())
        + "_"
        + original_name
    )

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.seek(0)

    file.save(upload_path)

    try:

        # ----------------------------------------
        # PREDICTION
        # ----------------------------------------

        result = predict_image(
            upload_path
        )

        prediction = result[0]

        confidence = result[1]

        probabilities = (
            result[2]
            if len(result) > 2
            else None
        )


        # ----------------------------------------
        # GRAD-CAM
        # ----------------------------------------

        gradcam_filename = (
            str(uuid.uuid4())
            + "_gradcam.jpg"
        )

        gradcam_path = os.path.join(
            app.config["RESULT_FOLDER"],
            gradcam_filename
        )

        try:

            make_gradcam(
                model=model,
                image_path=upload_path,
                output_path=gradcam_path
)

        except Exception as gradcam_error:

            print(
                "Grad-CAM error:",
                gradcam_error
            )

            gradcam_filename = None


        # ----------------------------------------
        # SAVE REPORT
        # ----------------------------------------

        probabilities_json = None

        if probabilities is not None:

            try:

                if hasattr(
                    probabilities,
                    "tolist"
                ):

                    probabilities = (
                        probabilities.tolist()
                    )

                probabilities_json = (
                    json.dumps(
                        probabilities
                    )
                )

            except Exception:

                probabilities_json = None


        report = Report(

            user_id=session[
                "user_id"
            ],

            image=filename,

            prediction=prediction,

            confidence=float(
                confidence
            ),

            probabilities=(
                probabilities_json
            ),

            gradcam=gradcam_filename,

            review_status="Pending"

        )

        db.session.add(report)

        db.session.commit()


        return redirect(
            url_for(
                "patient_report",
                report_id=report.id
            )
        )


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        if os.path.exists(
            upload_path
        ):

            os.remove(
                upload_path
            )

        return render_template(
            "error.html",
            message=(
                "Unable to process the MRI. "
                "Please check the image and try again."
            )
        ), 500


# ============================================================
# PATIENT REPORT
# ============================================================

@app.route(
    "/patient/report/<int:report_id>"
)
def patient_report(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    if session.get("role") != "patient":

        return "Access denied", 403

    report = Report.query.filter_by(
        id=report_id,
        user_id=session["user_id"]
    ).first_or_404()

    return render_template(
        "report_detail.html",
        report=report
    )


# ============================================================
# GENERAL REPORT VIEW
# ============================================================

@app.route(
    "/report/<int:report_id>"
)
def view_report(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    report = Report.query.get_or_404(
        report_id
    )

    # Patient can only view own report

    if session.get("role") == "patient":

        if report.user_id != session[
            "user_id"
        ]:

            return "Access denied", 403

    # Only doctor/admin otherwise

    elif session.get("role") not in [
        "doctor",
        "admin"
    ]:

        return "Access denied", 403

    return render_template(
        "report_detail.html",
        report=report
    )


# ============================================================
# REPORTS
# ============================================================

@app.route("/reports")
def reports():

    if not login_required():

        return redirect(
            url_for("login")
        )

    reports = Report.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        Report.id.desc()
    ).all()

    return render_template(
        "reports.html",
        reports=reports
    )


# ============================================================
# AI REPORT EXPLANATION
# ============================================================

@app.route(
    "/api/report/<int:report_id>/explain",
    methods=["POST"]
)
def ai_report_explanation(report_id):

    if not login_required():

        return jsonify({
            "error": "Login required"
        }), 401

    report = Report.query.get_or_404(
        report_id
    )

    # Patient access check

    if session.get("role") == "patient":

        if report.user_id != session[
            "user_id"
        ]:

            return jsonify({
                "error": "Access denied"
            }), 403

    elif session.get("role") not in [
        "doctor",
        "admin"
    ]:

        return jsonify({
            "error": "Access denied"
        }), 403

    try:

        explanation = explain_report(

            prediction=report.prediction,

            confidence=report.confidence,

            probabilities=report.probabilities

        )

        return jsonify({

            "success": True,

            "explanation": explanation

        })

    except Exception as e:

        print(
            "AI explanation error:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# AI REPORT CHAT
# ============================================================

@app.route(
    "/api/report/<int:report_id>/chat",
    methods=["POST"]
)
def report_ai_chat(report_id):

    if not login_required():

        return jsonify({
            "error": "Login required"
        }), 401

    report = Report.query.get_or_404(
        report_id
    )

    # Patient access

    if session.get("role") == "patient":

        if report.user_id != session[
            "user_id"
        ]:

            return jsonify({
                "error": "Access denied"
            }), 403

    elif session.get("role") not in [
        "doctor",
        "admin"
    ]:

        return jsonify({
            "error": "Access denied"
        }), 403

    data = request.get_json(
        silent=True
    ) or {}

    message = data.get(
        "message",
        ""
    ).strip()

    if not message:

        return jsonify({

            "error":
            "Message is required"

        }), 400

    try:

        answer = chat_about_report(

            question=message,

            prediction=report.prediction,

            confidence=report.confidence

        )

        return jsonify({

            "success": True,

            "answer": answer

        })

    except Exception as e:

        print(
            "AI chat error:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# GENERAL AI CHAT
# ============================================================

@app.route(
    "/api/ai-chat",
    methods=["POST"]
)
def ai_chat():

    if not login_required():

        return jsonify({

            "error":
            "Please login first."

        }), 401

    data = request.get_json(
        silent=True
    ) or {}

    message = data.get(
        "message",
        ""
    ).strip()

    if not message:

        return jsonify({

            "reply":
            "Please enter a question."

        }), 400

    try:

        reply = chat_with_ai(
            message
        )

        return jsonify({

            "reply": reply

        })

    except Exception as e:

        print(
            "AI chat error:",
            e
        )

        return jsonify({

            "reply":
            "Unable to contact the AI assistant."

        }), 500


# ============================================================
# AI EXPLANATION PAGE
# ============================================================

@app.route(
    "/explain-report/<int:report_id>"
)
def explain_report_page(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    report = Report.query.get_or_404(
        report_id
    )

    # Patient can only see own report

    if session.get("role") == "patient":

        if report.user_id != session[
            "user_id"
        ]:

            return "Access denied", 403

    elif session.get("role") not in [
        "doctor",
        "admin"
    ]:

        return "Access denied", 403

    try:

        explanation = explain_report(

            prediction=report.prediction,

            confidence=report.confidence,

            probabilities=report.probabilities

        )

    except Exception as e:

        explanation = (
            "AI explanation unavailable: "
            + str(e)
        )

    return render_template(

        "ai_explanation.html",

        report=report,

        explanation=explanation

    )


# ============================================================
# DOCTOR DASHBOARD
# ============================================================

@app.route(
    "/doctor-dashboard"
)
def doctor_dashboard():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if session.get("role") != "doctor":

        return "Access denied", 403

    user = User.query.get(
        session["user_id"]
    )

    reports = (
        db.session.query(
            Report,
            User.name.label(
                "patient_name"
            )
        )
        .join(
            User,
            Report.user_id == User.id
        )
        .order_by(
            Report.id.desc()
        )
        .all()
    )

    report_list = []

    for report, patient_name in reports:

        report.patient_name = (
            patient_name
        )

        report_list.append(
            report
        )

    pending_count = sum(

        1

        for report in report_list

        if report.review_status
        != "Reviewed"

    )

    reviewed_count = sum(

        1

        for report in report_list

        if report.review_status
        == "Reviewed"

    )

    return render_template(

        "doctor_dashboard.html",

        user=user,

        reports=report_list,

        pending_count=pending_count,

        reviewed_count=reviewed_count

    )


# ============================================================
# DOCTOR REVIEW
# ============================================================

@app.route(
    "/doctor/review/<int:report_id>",
    methods=["GET", "POST"]
)
def doctor_review(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    if session.get("role") != "doctor":

        return "Access denied", 403

    report = Report.query.get_or_404(
        report_id
    )

    patient = User.query.get(
        report.user_id
    )

    if request.method == "POST":

        report.review_status = (
            request.form.get(
                "review_status",
                "Pending"
            )
        )

        report.doctor_comment = (
            request.form.get(
                "doctor_comment",
                ""
            ).strip()
        )

        db.session.commit()

        return redirect(
            url_for(
                "doctor_dashboard"
            )
        )

    return render_template(

        "doctor_review.html",

        report=report,

        patient=patient

    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route(
    "/admin-dashboard"
)
def admin_dashboard():

    if session.get("role") != "admin":

        return redirect(
            url_for("login")
        )

    users = User.query.all()

    reports = Report.query.all()

    return render_template(

        "admin_dashboard.html",

        users=users,

        reports=reports

    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route(
    "/upload",
    methods=["GET"]
)
def upload():

    if not login_required():

        return redirect(
            url_for("login")
        )

    if session.get("role") != "patient":

        return "Access denied", 403

    return render_template(
        "upload.html"
    )


# ============================================================
# RESULT PAGE
# ============================================================

@app.route(
    "/result/<int:report_id>"
)
def result(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    report = Report.query.get_or_404(
        report_id
    )

    if session.get("role") == "patient":

        if report.user_id != session[
            "user_id"
        ]:

            return "Access denied", 403

    return render_template(

        "result.html",

        report=report

    )


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

@app.route(
    "/ai-assistant"
)
def ai_assistant():

    if not login_required():

        return redirect(
            url_for("login")
        )

    return render_template(
        "ai_assistant.html"
    )


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route(
    "/model-training"
)
def model_training():

    if session.get("role") != "admin":

        return redirect(
            url_for("login")
        )

    history_path = os.path.join(

        BASE_DIR,

        "static",

        "models",

        "training_history.json"

    )

    history = None

    if os.path.exists(
        history_path
    ):

        try:

            with open(
                history_path,
                "r"
            ) as file:

                history = json.load(
                    file
                )

        except Exception as e:

            print(
                "Training history error:",
                e
            )

    model_exists = os.path.exists(
        MODEL_PATH
    )

    return render_template(

        "model_training.html",

        history=history,

        model_exists=model_exists

    )


# ============================================================
# ANALYTICS
# ============================================================

@app.route(
    "/analytics"
)
def analytics():

    if not login_required():

        return redirect(
            url_for("login")
        )

    total_scans = Report.query.count()

    glioma_count = Report.query.filter_by(
        prediction="Glioma"
    ).count()

    meningioma_count = Report.query.filter_by(
        prediction="Meningioma"
    ).count()

    pituitary_count = Report.query.filter_by(
        prediction="Pituitary"
    ).count()

    no_tumor_count = Report.query.filter_by(
        prediction="No Tumor"
    ).count()

    return render_template(

        "analytics.html",

        total_scans=total_scans,

        glioma_count=glioma_count,

        meningioma_count=meningioma_count,

        pituitary_count=pituitary_count,

        no_tumor_count=no_tumor_count

    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# CONTACT
# ============================================================

@app.route(
    "/contact",
    methods=["GET", "POST"]
)
def contact():

    if request.method == "POST":

        return """
        <h2>Message sent successfully!</h2>
        <a href="/">Return Home</a>
        """

    return render_template(
        "contact.html"
    )


# ============================================================
# PDF REPORT
# ============================================================

@app.route(
    "/download-report/<int:report_id>"
)
def download_report(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    report = Report.query.get_or_404(
        report_id
    )

    # Patient can download own report

    if session.get("role") == "patient":

        if report.user_id != session[
            "user_id"
        ]:

            return "Access denied", 403

    elif session.get("role") not in [
        "doctor",
        "admin"
    ]:

        return "Access denied", 403

    user = User.query.get(
        report.user_id
    )

    filename = (
        f"brain_scan_report_"
        f"{report.id}.pdf"
    )

    output_path = os.path.join(

        app.config["RESULT_FOLDER"],

        filename

    )

    try:

        generate_pdf(

            report,

            user,

            output_path

        )

    except Exception as e:

        print(
            "PDF generation error:",
            e
        )

        return (
            "Unable to generate PDF.",
            500
        )

    return redirect(

        url_for(

            "static",

            filename=(
                f"results/{filename}"
            )

        )

    )


# ============================================================
# FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "File is too large. "
        "Maximum size is 10 MB.",
        413
    )


# ============================================================
# GENERAL ERROR
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    return render_template(

        "error.html",

        message=(
            "An internal server error "
            "occurred."
        )

    ), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="127.0.0.1",

        port=port,

        debug=True

    )