from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

engine = create_engine('sqlite:///model/job_postings.db')
df = pd.read_sql('SELECT * FROM job_postings', con=engine)
with open('model/tfidf_model.pkl', 'rb') as f:
    tfidf = pickle.load(f)

if 'combined_text' not in df.columns:
    df['combined_text'] = df['title'].fillna('') + " " + df['location'].fillna('') + " " + df['description'].fillna('') + df['work_type'].fillna('')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.casefold()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['preprocessed_text'] = df['combined_text'].apply(preprocess)
tfidf_matrix = tfidf.transform(df['preprocessed_text'])

def generate_confusion_matrix_img(conf_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Pred: Relevant", "Pred: Not Relevant"],
                yticklabels=["Actual: Relevant", "Actual: Not Relevant"], ax=ax)
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return encoded

# Dictionary kategori -> keywords pencarian di title
JOB_CATEGORY_KEYWORDS = {
    "Accounting": ["accountant", "accounting", "finance", "auditor", "bookkeeper", "tax", "payroll", "financial reporting", "controller", "cost accounting"],
    "Actuarial / Statistics": ["actuarial", "actuary", "statistician", "statistics", "data analyst", "risk analyst", "probability", "data modeling"],
    "Administration": ["admin", "administration", "administrative", "office assistant", "secretary", "clerical", "receptionist", "executive assistant", "office manager"],
    "Agency Digital": ["digital marketing", "social media", "seo", "content creator", "copywriter", "graphic designer", "ui designer", "ux designer", "media buyer", "branding", "marketing agency", "digital strategist", "performance marketing"],
    "Agriculture / Plantation": ["agriculture", "farmer", "plantation", "horticulture", "crop management", "farm worker", "agronomist"],
    "Architecture": ["architect", "architecture", "designer", "drafter", "cad", "urban planning", "landscape architect", "building design"],
    "Automotive": ["automotive", "mechanic", "car", "vehicle", "technician", "auto repair", "service technician", "auto electrician", "body shop"],
    "Banking": ["bank", "loan", "credit", "teller", "financial", "investment banking", "mortgage", "branch manager", "risk management"],
    "Biotechnology": ["biotech", "biotechnology", "researcher", "lab technician", "molecular biology", "genetics", "clinical research"],
    "Business Development": ["business development", "sales", "account manager", "client", "partnerships", "lead generation", "market development"],
    "Call Center": ["call center", "customer service", "telemarketer", "contact center", "inbound", "outbound", "call agent"],
    "Cleaning / Housekeeping": ["cleaning", "housekeeping", "janitor", "cleaner", "maid", "custodian", "sanitation"],
    "Construction": ["construction", "builder", "civil engineer", "site supervisor", "project engineer", "foreman", "contractor", "construction worker"],
    "Content Creator": ["content creator", "content writer", "copywriter", "blogger", "social media content", "video content", "editorial"],
    "Customer Relationship Management (CRM)": ["crm", "customer relationship", "client", "salesforce", "customer success", "client management"],
    "Customer Service": ["customer service", "support", "help desk", "call center", "client support", "technical support"],
    "Data Analyst": ["data analyst", "data analysis", "analyst", "data visualization", "excel", "sql", "power bi", "tableau"],
    "Data Scientist": ["data scientist", "machine learning", "ai", "artificial intelligence", "deep learning", "python", "r", "big data"],
    "Graphic Design": ["graphic designer", "graphic design", "illustrator", "photoshop", "adobe illustrator", "branding", "visual design"],
    "Digital Marketing": ["digital marketing", "seo", "social media", "advertising", "ppc", "google ads", "facebook ads", "content marketing"],
    "Driver / Courier": ["driver", "courier", "delivery", "logistics", "truck driver", "van driver", "dispatcher"],
    "E-commerce": ["e-commerce", "online shop", "marketplace", "shopify", "woocommerce", "product listing", "digital sales"],
    "Education": ["teacher", "education", "lecturer", "instructor", "tutor", "professor", "training", "curriculum development"],
    "Electrical Engineering": ["electrical engineer", "electronics", "technician", "circuit design", "power systems", "control systems"],
    "Engineering": ["engineer", "engineering", "mechanical engineer", "civil engineer", "electrical engineer", "design engineer", "project engineer"],
    "Event Management": ["event", "event organizer", "event management", "event planner", "conference", "trade show", "exhibition"],
    "Fashion / Apparel": ["fashion", "apparel", "designer", "fashion buyer", "garment", "textile", "stylist"],
    "Finance": ["finance", "financial analyst", "accountant", "investment", "budgeting", "forecasting", "controller", "financial planning"],
    "Food and Beverage": ["food", "beverage", "chef", "waiter", "restaurant", "barista", "kitchen staff", "food service"],
    "Geographic Information System (GIS)": ["gis", "geographic", "mapping", "cartography", "remote sensing", "spatial analysis", "geospatial"],
    "Graphic Motion / Animation": ["animation", "motion graphics", "animator", "after effects", "3d animation", "character design"],
    "Health & Medical": ["health", "medical", "doctor", "nurse", "physician", "healthcare", "medical assistant"],
    "Healthcare Support": ["healthcare support", "caregiver", "medical assistant", "home health aide", "nursing assistant"],
    "Help Desk / IT Support": ["help desk", "it support", "technical support", "desktop support", "system administration"],
    "Hospitality": ["hospitality", "hotel", "receptionist", "front desk", "concierge", "housekeeping"],
    "Human Resources": ["human resource", "hr", "recruitment", "talent acquisition", "payroll", "employee relations", "training and development"],
    "Information Technology": ["it", "information technology", "developer", "programmer", "software", "system analyst", "network admin", "database"],
    "Insurance": ["insurance", "underwriter", "claims", "broker", "risk management", "actuary"],
    "Interpreter / Translator": ["interpreter", "translator", "language", "linguist", "localization"],
    "Inventory Control": ["inventory", "stock", "warehouse", "inventory management", "stock control", "logistics"],
    "Journalism": ["journalist", "reporter", "editor", "news", "copy editor", "correspondent"],
    "Laboratory / Lab Technician": ["lab technician", "laboratory", "scientist", "lab assistant", "research technician"],
    "Legal": ["legal", "lawyer", "paralegal", "legal assistant", "compliance", "contract management"],
    "Logistics": ["logistics", "supply chain", "shipping", "freight", "warehouse management", "distribution"],
    "Maintenance / Technician": ["maintenance", "technician", "repair", "facility maintenance", "mechanical technician", "electrical maintenance"],
    "Manufacturing": ["manufacturing", "production", "factory", "assembly line", "quality control", "process improvement"],
    "Marketing": ["marketing", "brand", "promotion", "market research", "advertising", "public relations"],
    "Mechanic / Machine Technician": ["mechanic", "engineer", "machine", "technician", "machinist", "maintenance technician"],
    "Media & Communication": ["media", "communication", "public relations", "communications specialist", "corporate communications"],
    "Mining": ["mining", "miner", "geologist", "surveyor", "mine engineer"],
    "Network Engineer": ["network engineer", "network administrator", "network security", "systems engineer", "cisco", "routing", "switching"],
    "Operations": ["operations", "operation manager", "operations coordinator", "process management", "logistics"],
    "Photography": ["photographer", "photography", "photo editing", "studio photographer", "event photography"],
    "Procurement / Purchasing": ["procurement", "purchasing", "buyer", "sourcing", "supply management", "contract negotiation"],
    "Product Manager": ["product manager", "product owner", "product development", "roadmap", "agile"],
    "Project Management": ["project manager", "project coordinator", "scrum master", "program manager", "project planning"],
    "Quality Assurance": ["quality assurance", "qa tester", "tester", "software testing", "quality control", "automation testing"],
    "Real Estate / Property": ["real estate", "property", "broker", "property management", "leasing", "real estate agent"],
    "Receptionist / Front Office": ["receptionist", "front office", "front desk", "office receptionist", "administrative support"],
    "Research & Development": ["research", "r&d", "development", "scientific research", "innovation"],
    "Retail": ["retail", "sales associate", "store", "cashier", "merchandising", "customer service"],
    "Safety / Occupational Health and Safety (OHS)": ["safety", "health and safety", "ohs", "osha", "environmental health", "risk assessment"],
    "Sales": ["sales", "sales representative", "account manager", "business development", "client acquisition"],
    "Science / Research": ["scientist", "researcher", "lab", "experiment", "data analysis", "field research"],
    "Security / Guard": ["security", "guard", "security officer", "patrol", "surveillance"],
    "Shipping / Expedition": ["shipping", "expedition", "courier", "logistics", "freight forwarding"],
    "Social Media Specialist": ["social media", "social media specialist", "community manager", "content creator", "social media marketing"],
    "Software Developer": ["software developer", "programmer", "developer", "java", "python", "c#", "full stack", "backend", "frontend"],
    "Statistician": ["statistician", "statistics", "data analysis", "quantitative analysis", "modeling"],
    "Store Keeper / Warehouse": ["store keeper", "warehouse", "inventory", "stock management", "logistics", "materials handling"],
    "Supply Chain": ["supply chain", "logistics", "procurement", "inventory management", "demand planning"],
    "Teaching Assistant": ["teaching assistant", "ta", "classroom assistant", "education support"],
    "Technical Support": ["technical support", "help desk", "it support", "customer support", "desktop support"],
    "Telecommunications": ["telecommunications", "network", "telecom engineer", "fiber optics", "switching", "voip"],
    "Textile / Garment": ["textile", "garment", "fashion", "sewing", "pattern making", "quality control"],
    "Tourism / Travel Agent": ["tourism", "travel agent", "tour guide", "travel consultant", "customer service"],
    "Training & Development": ["training", "development", "trainer", "learning and development", "corporate training"],
    "UI/UX Designer": ["ui designer", "ux designer", "user interface", "user experience", "wireframing", "prototyping"],
    "Video Editing": ["video editor", "video editing", "post production", "adobe premiere", "final cut pro"],
    "Video Production": ["video production", "video producer", "cinematography", "film making", "camera operator"],
    "Warehouse Management": ["warehouse", "inventory", "logistics", "warehouse supervisor", "forklift operator"],
    "Web Developer": ["web developer", "web programmer", "frontend", "backend", "html", "css", "javascript", "react", "angular"],
    "Welding / Fabrication": ["welding", "fabrication", "welder", "mig welding", "tig welding", "arc welding", "metal fabrication"],
    "Writing / Editing": ["writing", "editor", "content writer", "copy editor", "proofreading", "technical writing"],
}

@app.route('/', methods=['GET', 'POST'])
def index():
    rekomendasi = []
    user_input = ""
    job_category = ""
    precision = recall = f1 = 0
    cmatrix = []
    evaluation = None
    img_conf_matrix = None
    total_jobs = 0

    if request.method == 'POST':
        job_category = request.form.get('job_category', '').strip()
        user_input = request.form.get('user_input', '').strip()
        experience = request.form.get('experience_level', '').strip()
        skills = request.form.get('skills', '').strip()
        education = request.form.get('education', '').strip()
        work_type = request.form.get('work_type', '').strip()

        # Filter berdasarkan kategori
        filtered_df = df.copy()
        if job_category:
            keywords = JOB_CATEGORY_KEYWORDS.get(job_category, [])
            if keywords:
                pattern = '|'.join([re.escape(k) for k in keywords])
                filtered_df = filtered_df[filtered_df['title'].str.contains(pattern, case=False, na=False)]

        # ⬇️ Tambahkan filter berdasarkan work_type (jika ada)
        if work_type:
            filtered_df = filtered_df[filtered_df['work_type'].str.lower() == work_type.lower()]

        if filtered_df.empty:
            return render_template("index.html",
                                   rekomendasi=[],
                                   user_input=user_input,
                                   job_category=job_category,
                                   evaluation=None,
                                   total_jobs=0,
                                   kategori_list=list(JOB_CATEGORY_KEYWORDS.keys()),
                                   img_conf_matrix=None,
                                   error_msg="No jobs found for the selected criteria.")

        # Gabungkan input pengguna
        full_input = f"{job_category} {user_input} {experience} {skills} {education} {work_type}".strip()

        # Jika semua input teks kosong (kecuali work_type), tampilkan 10 data saja
        if full_input.strip() == "":
            top_results = filtered_df.head(10)
            total_jobs = len(top_results)

            rekomendasi = top_results[['job_id', 'title', 'location', 'work_type', 'job_posting_url', 'description']].reset_index(drop=True)
            rekomendasi = rekomendasi.to_dict(orient='records')

            return render_template("index.html",
                                   rekomendasi=rekomendasi,
                                   user_input=user_input,
                                   job_category=job_category,
                                   evaluation=None,
                                   total_jobs=total_jobs,
                                   kategori_list=list(JOB_CATEGORY_KEYWORDS.keys()),
                                   img_conf_matrix=None)

        # Proses cosine similarity
        cleaned_input = preprocess(full_input)
        user_vector = tfidf.transform([cleaned_input])

        filtered_df = filtered_df.copy()
        filtered_df['preprocessed_text'] = filtered_df['combined_text'].apply(preprocess)
        filtered_tfidf_matrix = tfidf.transform(filtered_df['preprocessed_text'])
        cosine_sim = cosine_similarity(user_vector, filtered_tfidf_matrix)

        filtered_df['similarity_score'] = cosine_sim[0]

        relevan_results = filtered_df[filtered_df['similarity_score'] >= 0.1].sort_values(by='similarity_score', ascending=False).head(10)
        tidak_relevan_results = filtered_df[filtered_df['similarity_score'] < 0.1].sort_values(by='similarity_score', ascending=False).head(5)

        top_results = pd.concat([relevan_results, tidak_relevan_results])
        total_jobs = len(top_results)

        rekomendasi = top_results[['job_id', 'title', 'location', 'work_type', 'similarity_score', 'job_posting_url', 'description']].reset_index(drop=True)
        rekomendasi = rekomendasi.to_dict(orient='records')

        # Evaluasi manual
        keywords = JOB_CATEGORY_KEYWORDS.get(job_category, []) if job_category else []
        y_true = []
        y_pred = []

        for item in rekomendasi:
            title = item['title'].lower()
            is_relevant = any(kw in title for kw in keywords)
            y_true.append(1 if is_relevant else 0)
            y_pred.append(1 if item.get('similarity_score', 0) >= 0.1 else 0)

        if y_true and y_pred:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cmatrix = confusion_matrix(y_true, y_pred)
            img_conf_matrix = generate_confusion_matrix_img(cmatrix)

        evaluation = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'confusion_matrix': cmatrix.tolist() if len(cmatrix) else []
        }

    kategori_list = list(JOB_CATEGORY_KEYWORDS.keys())

    return render_template("index.html",
                           rekomendasi=rekomendasi,
                           user_input=user_input,
                           job_category=job_category,
                           evaluation=evaluation,
                           total_jobs=total_jobs,
                           kategori_list=kategori_list,
                           img_conf_matrix=img_conf_matrix)


@app.route('/evaluation', methods=['POST', 'GET'])
def evaluation_page():
    evaluation = None
    img_conf_matrix = None

    if request.method == 'POST':
        total = int(request.form.get('total', 0))
        y_true = []
        y_pred = []

        for idx in range(total):
            relevan = request.form.get(f'relevan_{idx}')
            sim_score = float(request.form.get(f'sim_{idx}', '0'))

            y_true.append(1 if relevan else 0)
            y_pred.append(1 if sim_score >= 0.1 else 0)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cmatrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
        img_conf_matrix = generate_confusion_matrix_img(cmatrix)

        evaluation = {
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1_score': round(f1, 2),
            'confusion_matrix': cmatrix
        }
    print(y_true)
    print(y_pred)
    return render_template("evaluation.html",
                           evaluation=evaluation,
                           img_conf_matrix=img_conf_matrix)

if __name__ == '__main__':
    app.run(debug=True)
