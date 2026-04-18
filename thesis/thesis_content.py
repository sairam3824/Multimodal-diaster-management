from __future__ import annotations


ABSTRACT = """
This project builds a disaster assessment system that combines three kinds of inputs already present in the repository: environmental metadata, social media image-text data, and post-disaster satellite images. The main goal was to avoid running these streams as separate tools and instead produce one fused result that can be reviewed in a web application.

The codebase has three branch models and one fusion block. The environmental branch reads a 32-value input vector. Those values come from weather, storm, seismic, and hydrological groups. The crisis branch processes one image and one text report. The satellite branch processes post-disaster xBD imagery. The outputs from these three branches are then sent to the `TriFusionLayer`.

The final system returns disaster type, severity, priority, population impact, and resource demand. According to the saved evaluation files in this repository, the full tri-modal setting reaches 99.41% priority accuracy and 0.0398 severity mean absolute error. The ablation results also show that the model can still run with missing modalities, and that the satellite branch gives the biggest gain over the crisis-only setting.

The project also includes Grad-CAM based explanation support, a FastAPI server, background analysis jobs, and a browser-based dashboard for running and reviewing cases. Overall, the work shows that the repository is an integrated prototype for multimodal disaster analysis rather than a collection of unrelated models.
""".strip()


ACKNOWLEDGEMENT = """
We thank {guide_name}, {guide_designation}, for guiding us throughout this senior design project. The regular discussions on model design, debugging, evaluation, and documentation helped us complete the work in a structured way.

We also thank the faculty and staff of {institution}, especially the {school}, for providing the environment and resources needed for this project. Their support helped us combine research work, software implementation, and documentation in one complete submission.

We are grateful to our classmates, friends, and family members for their support during training runs, testing, review, and final report preparation.
""".strip()


ACRONYMS = [
    ("API", "Application Programming Interface"),
    ("AUC", "Area Under the Receiver Operating Characteristic Curve"),
    ("BLIP", "Bootstrapping Language-Image Pre-training"),
    ("CNN", "Convolutional Neural Network"),
    ("CPU", "Central Processing Unit"),
    ("CrisisMMD", "Crisis Multimodal Twitter Dataset"),
    ("F1", "Harmonic Mean of Precision and Recall"),
    ("GIS", "Geographic Information System"),
    ("GPU", "Graphics Processing Unit"),
    ("IoT", "Internet of Things"),
    ("MAE", "Mean Absolute Error"),
    ("MHA", "Multi-Head Attention"),
    ("ROC", "Receiver Operating Characteristic"),
    ("ViT", "Vision Transformer"),
    ("XAI", "Explainable Artificial Intelligence"),
    ("xBD", "xView2 Building Damage Assessment Dataset"),
    ("XLM-R", "XLM-RoBERTa"),
]


CHAPTERS = [
    {
        "title": "CHAPTER 1 INTRODUCTION",
        "elements": [
            {
                "type": "section",
                "title": "Background and Motivation",
                "paragraphs": [
                    "During a disaster, the available information is usually split across different sources. A social media post may show what people are seeing. A satellite image may show physical damage over a large area. Environmental or seismic values may show the hazard condition itself. In many cases, these sources are viewed separately even though they describe the same event.",
                    "This project started from that problem. In our repository, the environmental model, the crisis-media model, and the satellite model were all useful on their own, but each one answered only part of the question. We wanted one system that could use all three and produce a single operational result instead of three separate outputs.",
                    "The result is the Multimodal Disaster Intelligence Platform. It combines environmental metadata, image-text social reports, and satellite imagery through a tri-modal fusion layer. The aim was not only to improve prediction scores, but also to make the output easier to review through explanations, saved case history, and a working browser interface."
                ]
            },
            {
                "type": "section",
                "title": "Problem Statement",
                "paragraphs": [
                    "The main problem addressed in this project is how to combine disaster evidence from different modalities into one usable assessment. A person using the system should not have to read separate outputs from a sensor model, a social media model, and a satellite model and then mentally combine them.",
                    "A second problem is missing data. In real use, a case may begin with only a social image and short text. Satellite data may come later. Sensor values may be partial. So the system should still return a useful result even when not all inputs are present."
                ]
            },
            {
                "type": "section",
                "title": "Objectives",
                "paragraphs": [
                    "The first objective was to build a single pipeline for tri-modal disaster assessment. The system should identify disaster type, estimate severity, assign a priority level, and provide resource-related outputs from the available evidence.",
                    "The second objective was to make the system usable as an application and not just as a set of training notebooks. For that reason, the project also includes an API server, background analysis jobs, explanation support, and a five-page web dashboard for running and reviewing cases."
                ]
            },
            {
                "type": "section",
                "title": "Scope of the Project",
                "paragraphs": [
                    "In the current repository, the disaster classes used in training and evaluation are fire, storm, earthquake, and flood. The folder name `IOT/` is kept from earlier development, but the present environmental branch is trained from archived datasets and not from live field hardware.",
                    "The work completed in this project includes data preparation, branch-model training, fusion training, evaluation, explanation output, API serving, and the web interface. Field deployment, agency integration, and testing on a naturally aligned real-world tri-modal dataset were not part of the current implementation."
                ]
            },
            {
                "type": "section",
                "title": "Key Contributions",
                "paragraphs": [
                    "The main contribution of this work is the tri-modal fusion pipeline that joins environmental metadata, crisis-media features, and satellite features inside one decision layer instead of treating them as separate tasks.",
                    "Another contribution is the adaptive handling of modalities and sensor groups. The system learns which environmental groups matter for a case and can still work when satellite or environmental inputs are missing.",
                    "The project also contributes a usable software layer on top of the models. The repository includes explanation logic, a FastAPI server, background job handling, saved-case review, and a web interface for running the system end to end."
                ]
            },
            {
                "type": "figure",
                "path": "architecture.png",
                "caption": "Overall architecture of the Multimodal Disaster Intelligence Platform showing the three upstream branches, the fusion layer, and the downstream decision heads.",
                "width_inches": 6.2
            },
            {
                "type": "section",
                "title": "Organization of the Report",
                "paragraphs": [
                    "Chapter 2 summarizes related work that is relevant to our design. Chapter 3 describes the system structure and datasets. Chapter 4 explains the proposed method. Chapter 5 covers implementation and the web application. Chapter 6 presents the experimental results. Chapter 7 closes the report with limitations and future work."
                ]
            }
        ]
    },
    {
        "title": "CHAPTER 2 LITERATURE REVIEW",
        "elements": [
            {
                "type": "section",
                "title": "Social Media for Crisis Understanding",
                "paragraphs": [
                    "A large part of early crisis-analysis work used only text. These systems classified tweets, short reports, or event-related messages, but they did not make use of the attached images.",
                    "Later work, especially on CrisisMMD, showed that image and text together usually perform better than either one alone. That idea is directly relevant to our crisis branch. However, most of those systems stop at social-media classification and do not combine the result with environmental evidence or satellite damage information."
                ]
            },
            {
                "type": "section",
                "title": "Satellite Damage Assessment",
                "paragraphs": [
                    "Satellite imagery is commonly used for post-disaster damage mapping because it covers a large area in one view. Benchmarks such as xBD and xView2 made building-damage segmentation a standard task, and models such as DeepLab and U-Net are widely used for it.",
                    "In our project, the satellite branch is not used only as a standalone segmenter. It is also used as a feature generator for the fusion layer. That is an important difference. Even if pixel-level results are not perfect on every class, the satellite embedding can still improve the final fused decision."
                ]
            },
            {
                "type": "section",
                "title": "Environmental and Sensor-Based Hazard Monitoring",
                "paragraphs": [
                    "Environmental monitoring has long been used for floods, storms, and earthquakes. Rainfall, pressure, wind, seismic depth, and similar values can show the physical state of a disaster much earlier than public reports.",
                    "The weakness of many hazard-monitoring systems is that they are built for one task at a time. They also often assume that every sensor group matters equally. Our environmental branch takes a different approach by grouping the features and learning which group should get more weight for a given input."
                ]
            },
            {
                "type": "section",
                "title": "Multimodal Fusion and Cross-Attention",
                "paragraphs": [
                    "A basic multimodal model usually joins feature vectors by concatenation. Newer models use attention or cross-attention so that one input can affect how another input is interpreted. This is useful in disaster analysis because image, text, and environmental data do not contribute equally in every case.",
                    "Most work in this area still focuses on only two modalities at once. Our project extends that idea to three branches and combines them through pairwise cross-attention and a learned gate. Missing modalities are also handled during training so the system can still run when all inputs are not available."
                ]
            },
            {
                "type": "section",
                "title": "Explainability in High-Stakes AI Systems",
                "paragraphs": [
                    "A disaster-response system should not return only a label or score. The user also needs some explanation of how that result was produced. Visual explanation methods such as Grad-CAM help by showing which region of an image influenced the prediction.",
                    "In this project, explanation is part of the normal workflow. The incident page shows heatmaps, class probabilities, modality weights, and sensor-group weights along with the final result."
                ]
            },
            {
                "type": "section",
                "title": "Research Gap",
                "paragraphs": [
                    "The main gap is that many existing systems solve only one part of the disaster-analysis problem. Social media models focus on public reports. Satellite models focus on damage mapping. Sensor systems focus on physical measurements. Fewer systems try to combine all of them in one pipeline and then expose the result through a usable interface.",
                    "This project addresses that gap by combining model training, fusion, explanation, API serving, and a web application in one repository. That is the part that separates it from a small benchmark experiment."
                ]
            }
        ]
    },
    {
        "title": "CHAPTER 3 SYSTEM DESIGN AND DATA PREPARATION",
        "elements": [
            {
                "type": "section",
                "title": "System Overview",
                "paragraphs": [
                    "The system has three main branches and one fusion stage. One branch handles environmental values, one handles crisis image-text input, and one handles satellite imagery. Each branch produces an embedding, and those embeddings are sent to the tri-fusion layer.",
                    "This layout made the project easier to build and test. We could work on one branch without changing the others, and the serving layer only had to combine prepared vectors instead of raw files."
                ]
            },
            {
                "type": "section",
                "title": "Modalities Used in the Project",
                "paragraphs": [
                    "The environmental branch uses a 32-value feature vector split into four groups: weather, storm, seismic, and hydrological values. The crisis branch takes one image and one paired text report. The satellite branch takes post-disaster satellite imagery from the xBD pipeline.",
                    "These three modalities were selected because they describe different parts of the same problem. Environmental data describes the hazard condition, social media describes what people are reporting, and satellite imagery shows visible damage over a wider area."
                ]
            },
            {
                "type": "table",
                "title": "Table 3.1 Complementary role of each modality in the proposed platform",
                "headers": ["Modality", "Primary strength", "Typical weakness", "Contribution inside fusion"],
                "rows": [
                    ["Environmental metadata", "Captures physical hazard signatures and contextual risk", "Limited human impact detail", "Improves hazard discrimination and context"],
                    ["Social media image-text", "Fast public-facing situation updates", "Noisy, incomplete, and credibility-sensitive", "Adds on-the-ground impact evidence"],
                    ["Satellite imagery", "Wide-area structural damage visibility", "May arrive later and miss community context", "Strong signal for severity and priority"],
                ]
            },
            {
                "type": "section",
                "title": "Environmental Dataset Preparation",
                "paragraphs": [
                    "The code still uses the older `IOT` name, but the current environmental branch was trained from archived datasets and not from live devices. In this repository, the main sources are fire-weather records, tropical storm tracks, earthquake catalogs, and a flood-risk dataset. These inputs were converted into one shared feature format before training.",
                    "After preprocessing, the environmental dataset used in the repository contains 63,527 samples. The final labels are fire, storm, earthquake, flood, and unknown. A shared input format was needed because the original source files used different columns, scales, and naming styles."
                ]
            },
            {
                "type": "table",
                "title": "Table 3.2 Environmental metadata sources used in the project",
                "headers": ["Source dataset", "Mapped hazard class", "Rows used"],
                "rows": [
                    ["CA Wildfire weather and fire records", "Fire", "14,988"],
                    ["Historical Tropical Storm Tracks", "Storm", "59,228"],
                    ["Atlantic storm records", "Storm", "22,705"],
                    ["Global earthquake catalog", "Earthquake", "8,394"],
                    ["Iran earthquake catalog", "Earthquake", "52,268"],
                    ["Sri Lanka flood risk dataset", "Flood", "25,000"],
                ]
            },
            {
                "type": "section",
                "title": "Crisis and Satellite Datasets",
                "paragraphs": [
                    "The crisis branch uses CrisisMMD. This dataset contains paired images and short text messages with humanitarian labels. It fits this project because the branch is designed to read image and text together.",
                    "The satellite branch uses xBD-based post-disaster imagery. In this project, that branch is used both for damage understanding and for generating embeddings that can be used in fusion."
                ]
            },
            {
                "type": "section",
                "title": "Feature Engineering Strategy",
                "paragraphs": [
                    "The environmental features are normalized to a common range and arranged into groups by meaning. Time-related values such as month or hour are encoded cyclically. Some values are inverted where needed so that larger normalized values match higher disaster relevance more consistently.",
                    "The grouped format is important for the later weighting step. Because weather, storm, seismic, and hydro values stay in separate blocks, the model can learn which block matters more for a given sample."
                ]
            },
            {
                "type": "section",
                "title": "Problem Formulation",
                "paragraphs": [
                    "The fusion stage is treated as a multi-task problem. For each case, the system predicts disaster type, severity, priority, population impact, and resource demand instead of only one class label.",
                    "The three upstream branches do not produce outputs of the same size. The environmental branch outputs a 128-dimensional vector, the crisis branch outputs a 1024-dimensional vector, and the satellite branch outputs a 640-dimensional vector. These are projected into a shared space before fusion."
                ]
            }
        ]
    },
    {
        "title": "CHAPTER 4 PROPOSED METHODOLOGY",
        "elements": [
            {
                "type": "section",
                "title": "Environmental Branch: AdaptiveIoTClassifier",
                "paragraphs": [
                    "The environmental branch is built around four sensor groups. Each group is encoded separately first, and each 8-value group is mapped into a 128-dimensional hidden representation.",
                    "The branch also includes a `SensorConfidenceEstimator` for each group. Instead of giving equal weight to every group, the model estimates which group is more useful for the current sample and uses that score during weighting.",
                    "After that weighting step, the four group embeddings pass through cross-group attention. The pooled result is then used for disaster type, severity, and related outputs."
                ]
            },
            {
                "type": "figure",
                "path": "outputs-paper/iot/sensor_group_weights.png",
                "caption": "Learned sensor-group importance across disaster categories in the environmental branch. The model assigns high weight to the physically relevant feature families for each hazard type.",
                "width_inches": 5.8
            },
            {
                "type": "section",
                "title": "Crisis Branch: AdaptiveFusionClassifier",
                "paragraphs": [
                    "The crisis branch combines image and text from a social media post. The image is processed through a BLIP-based vision encoder, and the text is processed through XLM-RoBERTa. Both outputs are projected before fusion.",
                    "This branch also learns separate confidence values for image and text. That means the model does not treat both inputs as equally useful for every sample.",
                    "Image-to-text and text-to-image attention are both used. The attended outputs are concatenated and passed forward as the crisis embedding."
                ]
            },
            {
                "type": "section",
                "title": "Satellite Branch: Modified DeepLabV3+",
                "paragraphs": [
                    "The satellite branch uses a modified DeepLabV3+ model. In this project, it is not used only for segmentation. It is also used to produce an embedding for the fusion layer.",
                    "The branch outputs both damage-related spatial information and a compact scene representation. The final satellite embedding size used in fusion is 640.",
                    "The experimental results show that this branch still has room for improvement on minority damage classes. Even so, it contributes strongly to the final fused prediction."
                ]
            },
            {
                "type": "section",
                "title": "Tri-Fusion Layer",
                "paragraphs": [
                    "The fusion layer first projects the environmental, crisis, and satellite embeddings into one shared hidden size. This is needed because the three branches do not produce vectors of the same size.",
                    "After projection, the model applies three pairwise cross-attention steps: crisis with environment, crisis with satellite, and environment with satellite.",
                    "A modality gate is then used to weight the branches before the final shared output is formed. If a modality is missing, the model uses its fallback representation and masks the missing branch during gating."
                ]
            },
            {
                "type": "table",
                "title": "Table 4.1 Core configuration of the tri-fusion model",
                "headers": ["Parameter", "Value"],
                "rows": [
                    ["Fusion model", "TriFusionLayer with pairwise cross-attention and adaptive gating"],
                    ["Projected hidden dimension", "256"],
                    ["Crisis embedding size", "1024"],
                    ["Environmental embedding size", "128"],
                    ["Satellite embedding size", "640"],
                    ["Total parameters", "3,163,602"],
                    ["Training split", "10,886 train / 2,722 validation"],
                    ["Modality dropout", "30% on environmental and satellite branches"],
                    ["Optimizer", "AdamW with cosine annealing"],
                    ["Best validation loss", "0.0288 at epoch 15"],
                ]
            },
            {
                "type": "section",
                "title": "Multi-Task Prediction Heads",
                "paragraphs": [
                    "The shared fusion output is connected to several prediction heads. These heads produce priority, disaster type, severity, population impact, and resource-demand outputs.",
                    "This setup was chosen because the project is not only a hazard-classification task. The final application also needs urgency and support-related outputs."
                ]
            },
            {
                "type": "section",
                "title": "Explainability and User-Facing Reasoning",
                "paragraphs": [
                    "After prediction, the system also generates explanation outputs. For image-based inputs, Grad-CAM style visual outputs are used to show important regions.",
                    "The project also uses fallback logic so that explanation output is still available even when one preferred path does not work for a given encoder or input."
                ]
            }
        ]
    },
    {
        "title": "CHAPTER 5 IMPLEMENTATION AND SYSTEM INTEGRATION",
        "elements": [
            {
                "type": "section",
                "title": "Repository Organization",
                "paragraphs": [
                    "The repository is split by major component. `IOT/` contains the environmental branch. `crisis/` contains the social-media branch. `XBD/` contains the satellite notebooks and related files. `fusion/` contains the integration code, the FastAPI server, the explanation utilities, and the web application.",
                    "This layout follows the way the project was built. Each branch was trained and checked separately before being connected through the shared inference pipeline."
                ]
            },
            {
                "type": "section",
                "title": "Training Workflow",
                "paragraphs": [
                    "Training was done in stages. The environmental branch, crisis branch, and satellite branch were prepared separately first. After that, the tri-fusion layer was trained using the saved branch outputs.",
                    "This approach kept the work manageable. It was easier to debug one branch at a time than to train the whole tri-modal system from raw inputs in a single step."
                ]
            },
            {
                "type": "section",
                "title": "Inference Pipeline",
                "paragraphs": [
                    "At inference time, the system accepts a crisis image and text, with optional satellite input and optional environmental values. The server runs the branches for which input is available and then sends those outputs to the fusion layer.",
                    "The main serving code is in `fusion/server.py`. The project supports both direct analysis and background jobs through `/analysis/jobs`. Uploaded files are written to temporary storage, and the job state is saved so completed or failed runs can still be checked later."
                ]
            },
            {
                "type": "table",
                "title": "Table 5.1 Major implementation components in the repository",
                "headers": ["Path", "Role in the system"],
                "rows": [
                    ["IOT/train_iot.py", "Training pipeline for the environmental metadata classifier"],
                    ["IOT/evaluate_iot.py", "Evaluation scripts and metric generation for the environmental branch"],
                    ["crisis/train_crisis_code.ipynb", "Training workflow for the social-media image-text model"],
                    ["XBD/final-xbd-deeplabv3plus.ipynb", "Satellite damage model training workflow"],
                    ["fusion/train_tri_fusion.py", "Training script for the tri-modal fusion layer"],
                    ["fusion/pipeline.py", "End-to-end prediction assembly across modalities"],
                    ["fusion/server.py", "FastAPI application exposing model inference"],
                    ["fusion/static/", "Web dashboard pages, styles, and JavaScript"],
                ]
            },
            {
                "type": "table",
                "title": "Table 5.2 Main web routes and their implementation roles",
                "headers": ["Route", "Primary function", "Implementation detail"],
                "rows": [
                    ["/overview", "Landing dashboard for latest system state", "Summarizes current alert, detected hazard, priority, latest update, and recent saved runs"],
                    ["/analysis", "Case submission workspace", "Collects crisis image, report text, optional satellite image, and optional environmental values before scheduling analysis"],
                    ["/incident", "Detailed fused result page", "Displays alert summary, crisis probabilities, modality weights, sensor weights, resource demand, and Grad-CAM outputs"],
                    ["/iot-monitor", "Environmental branch inspection page", "Shows hazard-specific IoT outputs such as fire probability, storm category proxy, earthquake magnitude proxy, flood risk, and sensor-group weights"],
                    ["/reports", "Saved history and export page", "Reads stored incident runs from browser history and exports them as JSON for later review"],
                ]
            },
            {
                "type": "section",
                "title": "Dashboard and Human Interaction Layer",
                "paragraphs": [
                    "The project includes a five-page dashboard inside `fusion/static/`. The main pages are overview, analysis, incident details, IoT monitor, and reports.",
                    "The client also stores history in the browser through `fusion/static/js/store.js`. Because of that, a user can reopen saved cases and export the stored report history from the reports page."
                ]
            },
            {
                "type": "section",
                "title": "Web Application Workflow",
                "paragraphs": [
                    "The web application is built around a simple workflow. The user starts from the overview page, opens the new-analysis form, submits the case, and then reviews the result on the incident page.",
                    "The analysis form can send the request as a background job through `/analysis/jobs`. While the job is running, the incident page shows a waiting state. When the result is ready, the saved record can be opened normally.",
                    "The incident page shows the fused result, probabilities, modality weights, sensor weights, and Grad-CAM outputs. The reports page stores earlier runs and allows export."
                ]
            },
            {
                "type": "figure",
                "path": "thesis/web_images/home_page.png",
                "caption": "Overview page of the web application. It summarizes the latest alert state, detected hazard, response priority, recent incident records, and the recommended navigation flow across the platform.",
                "width_inches": 6.2
            },
            {
                "type": "figure",
                "path": "thesis/web_images/analysis_1_page.png",
                "caption": "New-analysis page, first section. The operator can upload crisis imagery, enter the public report text, and optionally add satellite evidence before running the fused analysis.",
                "width_inches": 6.2
            },
            {
                "type": "figure",
                "path": "thesis/web_images/analysis_2_page.png",
                "caption": "New-analysis page, second section. The form accepts geographic context and environmental measurements so the same workflow can combine crisis, satellite, and environmental inputs in one submission.",
                "width_inches": 6.2
            },
            {
                "type": "figure",
                "path": "thesis/web_images/incident_page.png",
                "caption": "Incident-details page. This screen presents the fused alert level, hazard category, severity, modality contributions, crisis probabilities, sensor weights, and estimated resource demand for the active case.",
                "width_inches": 6.2
            },
            {
                "type": "figure",
                "path": "thesis/web_images/reports_page.png",
                "caption": "Reports page. Saved incident runs can be reviewed later, compared, and exported, which makes the application useful beyond one-time inference.",
                "width_inches": 6.2
            },
            {
                "type": "section",
                "title": "Engineering Trade-Offs",
                "paragraphs": [
                    "Some parts of the project were trained under limited hardware conditions, especially in the xBD and fusion stages. That affected how far the experiments could be pushed.",
                    "The codebase also keeps the older IoT naming even though the current environmental branch is trained from archival data. The name was kept to avoid breaking the existing pipeline and files.",
                    "For the web layer, the project uses browser-side history storage and a lightweight server-side job store instead of a full database."
                ]
            }
        ]
    },
    {
        "title": "CHAPTER 6 EXPERIMENTAL RESULTS AND DISCUSSION",
        "elements": [
            {
                "type": "section",
                "title": "Evaluation Strategy",
                "paragraphs": [
                    "The evaluation is reported at two levels. First, each branch model is evaluated on its own task. Second, the fused system is evaluated through ablation.",
                    "This makes it easier to see what each branch contributes and how the final tri-modal result changes when one modality is added or removed."
                ]
            },
            {
                "type": "section",
                "title": "Environmental Branch Results",
                "paragraphs": [
                    "According to the saved evaluation outputs, the environmental classifier reaches 97.64% overall accuracy, a macro F1 near 0.90, and a macro ROC AUC above 0.996. Storm, earthquake, and flood are separated very clearly in these results, while fire and unknown are harder to separate.",
                    "The severity head reports a mean absolute error close to 0.045 and an R-squared value around 0.745.",
                    "The sensor-group weights are also consistent with the task. Storm values dominate storm cases, seismic values dominate earthquake cases, and hydrological values dominate flood cases."
                ]
            },
            {
                "type": "figure",
                "path": "outputs-paper/iot/confusion_matrix.png",
                "caption": "Confusion matrix for the environmental metadata classifier. Storm, earthquake, and flood classes are strongly separated, while fire and unknown show the main overlap.",
                "width_inches": 5.6
            },
            {
                "type": "figure",
                "path": "outputs-paper/iot/roc_curves.png",
                "caption": "Per-class ROC curves for the environmental metadata classifier, showing strong class separation across most hazard categories.",
                "width_inches": 5.6
            },
            {
                "type": "figure",
                "path": "outputs-paper/iot/severity_regression.png",
                "caption": "Severity regression analysis for the environmental branch, including agreement between predicted and target severity values and the residual pattern.",
                "width_inches": 5.6
            },
            {
                "type": "section",
                "title": "Crisis Branch Results",
                "paragraphs": [
                    "The crisis-media model achieves 86.39% test accuracy and 87.48% validation F1 on the humanitarian classification task. That is a strong result for a branch that is also optimized to produce useful fusion embeddings rather than pure standalone accuracy. The class-wise results indicate that the model is especially strong on infrastructure damage, not-humanitarian content, and rescue-or-donation categories.",
                    "The weakest class is affected individuals, which is unsurprising given its lower support and the subtle overlap it can have with other humanitarian categories. Even so, the overall ROC behavior is strong, and the training history shows that the dual-modality weighting remains balanced rather than collapsing fully to either image or text."
                ]
            },
            {
                "type": "table",
                "title": "Table 6.1 Crisis branch per-class performance on the humanitarian task",
                "headers": ["Class", "Precision", "Recall", "F1 score", "AUC"],
                "rows": [
                    ["Affected individuals", "0.36", "0.56", "0.43", "0.96"],
                    ["Infrastructure damage", "0.87", "0.90", "0.88", "0.98"],
                    ["Not humanitarian", "0.90", "0.87", "0.89", "0.96"],
                    ["Other relevant information", "0.85", "0.86", "0.86", "0.97"],
                    ["Rescue, volunteering, or donation effort", "0.79", "0.83", "0.81", "0.98"],
                    ["Macro average", "-", "-", "0.774", "0.97"],
                ]
            },
            {
                "type": "figure",
                "path": "outputs-paper/crisis/training_history.png",
                "caption": "Training behavior of the crisis-media branch, showing stable validation performance and balanced confidence between visual and textual streams.",
                "width_inches": 5.8
            },
            {
                "type": "figure",
                "path": "outputs-paper/crisis/confusion_matrix.png",
                "caption": "Confusion matrix for the crisis-media branch on the humanitarian classification task.",
                "width_inches": 5.6
            },
            {
                "type": "section",
                "title": "Satellite Branch Results",
                "paragraphs": [
                    "The satellite branch reaches a best validation mean IoU of 0.4276 with mean F1 of 0.5221 in the checked-in experiments. The no-damage class performs very strongly, while the minority damage classes remain more difficult. This gap is consistent with the class imbalance known in xBD-style datasets.",
                    "From a standalone segmentation perspective, the branch still has headroom. However, the more important question for this project is whether the satellite embedding improves the final tri-modal system. The fusion ablation clearly shows that it does. Satellite evidence is the single strongest contributor to improved priority and severity prediction in the combined model."
                ]
            },
            {
                "type": "table",
                "title": "Table 6.2 Satellite branch class-wise validation performance",
                "headers": ["Damage class", "IoU", "F1 score"],
                "rows": [
                    ["No damage", "0.9774", "0.9885"],
                    ["Minor damage", "0.1595", "0.2662"],
                    ["Major damage", "0.2647", "0.3936"],
                    ["Destroyed", "0.3087", "0.4402"],
                    ["Mean", "0.4276", "0.5221"],
                ]
            },
            {
                "type": "figure",
                "path": "outputs-paper/xbd/training_history (1).png",
                "caption": "Training history of the satellite branch, including loss, mean IoU progression, and learning-rate scheduling.",
                "width_inches": 5.8
            },
            {
                "type": "section",
                "title": "Tri-Fusion Ablation Results",
                "paragraphs": [
                    "The ablation study shows what each modality adds to the final system. The crisis-only setting reaches 68.96% priority accuracy. Adding environmental data raises that to 72.37%. Adding satellite data raises it to 98.75%.",
                    "The full tri-modal setting reaches 99.41% priority accuracy. Severity mean absolute error drops from 0.1165 in crisis-only mode to 0.0398 in the full setting.",
                    "Across these settings, the results improve as more modalities are added. In the saved outputs, the satellite branch gives the largest single gain."
                ]
            },
            {
                "type": "table",
                "title": "Table 6.3 Ablation study across modality configurations",
                "headers": ["Configuration", "Severity MAE", "Priority accuracy", "Disaster accuracy", "Population MAE", "Resource MAE"],
                "rows": [
                    ["Crisis only", "0.1165", "68.96%", "100.00%", "0.0877", "0.0468"],
                    ["Crisis + environmental metadata", "0.0989", "72.37%", "100.00%", "0.0691", "0.0393"],
                    ["Crisis + satellite", "0.0415", "98.75%", "100.00%", "0.0274", "0.0164"],
                    ["Crisis + environmental metadata + satellite", "0.0398", "99.41%", "100.00%", "0.0248", "0.0161"],
                ]
            },
            {
                "type": "section",
                "title": "Robustness and Practical Observations",
                "paragraphs": [
                    "Repository notes also include noise-injection tests for the environmental branch. Under moderate Gaussian noise, the results drop gradually instead of failing immediately.",
                    "Another point from the saved experiments is that disaster-type accuracy is already saturated across the fusion settings. Because of that, severity and priority are the more useful metrics for comparing configurations."
                ]
            },
            {
                "type": "section",
                "title": "Limitations",
                "paragraphs": [
                    "The project has several limitations. The environmental branch is trained on archival datasets rather than live field devices. The tri-modal fusion stage also uses synthetic pairing because a naturally aligned dataset is not available in the repository.",
                    "The satellite branch is still weaker on the minority damage classes than current state-of-the-art segmentation systems, and some of the training was done under limited hardware conditions."
                ]
            }
        ]
    },
    {
        "title": "CHAPTER 7 CONCLUSION AND FUTURE WORK",
        "elements": [
            {
                "type": "section",
                "title": "Conclusion",
                "paragraphs": [
                    "This project resulted in a working multimodal disaster assessment prototype. It combines environmental data, crisis-media input, and satellite imagery in one pipeline and presents the fused result through the web interface.",
                    "The repository now includes the branch models, fusion code, explanation outputs, API layer, saved-case handling, and dashboard pages. In the recorded results, the full tri-modal setup performs better than the crisis-only setting, especially on priority and severity.",
                    "So the final outcome is more than a trained model. It is a runnable prototype that can be improved further in later work."
                ]
            },
            {
                "type": "section",
                "title": "Future Work",
                "paragraphs": [
                    "The first next step is to test the environmental branch on live or near-real-time sensor streams instead of only archived datasets. That would show how the current weighting logic behaves under noisy values, missing readings, and timing gaps.",
                    "Another important extension is better satellite training. Higher-resolution runs, improved class balancing, and stronger hardware support should help the minority damage classes and may further improve the fusion branch.",
                    "A longer-term goal is to evaluate the model on naturally aligned tri-modal disaster data. That would remove the current need for synthetic pairing and would give a better picture of how the full system behaves on real events."
                ]
            }
        ]
    }
]


REFERENCES = [
    "Abavisani, M., Wu, L., Hu, S., Tetreault, J., & Jaimes, A. (2020). Multimodal categorization of crisis events in social media. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.",
    "Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter datasets from natural disasters. In Proceedings of the International AAAI Conference on Web and Social Media.",
    "Allen, R. M. (2009). Real-time earthquake detection and hazard assessment by ElarmS across California. Geophysical Research Letters, 36(5).",
    "Basha, E., & Rus, D. (2008). Design of early warning flood detection systems for developing countries. In Proceedings of the International Conference on Information and Communication Technologies and Development.",
    "Centre for Research on the Epidemiology of Disasters. (2024). 2023 disasters in numbers. CRED/UCLouvain.",
    "Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.",
    "CORGIS Datasets Project. (2023). Earthquakes dataset. Virginia Tech.",
    "Gupta, R., Hosfelt, R., Saber, S., Ashton, N., Mullan, R., Campbell, S., et al. (2019). xBD: A dataset for assessing building damage from satellite imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.",
    "Imran, M., Castillo, C., Diaz, F., & Vieweg, S. (2015). Processing social media messages in mass emergency: A survey. ACM Computing Surveys, 47(4), 1-38.",
    "Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the International Conference on Machine Learning.",
    "Mouzannar, H., Rizk, Y., & Awad, M. (2025). Guided cross-attention for multimodal crisis classification with LLaVA-generated captions. Information Processing and Management, 62.",
    "Ofli, F., Alam, F., & Imran, M. (2020). Analysis of social media data using multimodal deep learning for disaster response. In Proceedings of ISCRAM.",
    "Olteanu, A., Castillo, C., Diaz, F., & Vieweg, S. (2014). CrisisLex: A lexicon for collecting and filtering microblogged communications in crises. In Proceedings of the International AAAI Conference on Web and Social Media.",
    "Rudner, T. G. J., Russwurm, M., Fil, J., Pelich, R., Bischke, B., Kopackova, V., & Bilinski, P. (2019). Multi3Net: Segmenting flooded buildings via fusion of multiresolution, multisensor, and multitemporal satellite imagery. Proceedings of the AAAI Conference on Artificial Intelligence.",
    "Sakaki, T., Okazaki, M., & Matsuo, Y. (2010). Earthquake shakes Twitter users: Real-time event detection by social sensors. In Proceedings of the 19th International Conference on World Wide Web.",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems.",
]


APPENDICES = [
    {
        "title": "APPENDIX A REPOSITORY SUMMARY",
        "paragraphs": [
            "The project repository is organized around the environmental, crisis, satellite, and fusion subsystems. Evaluation outputs and publication-oriented material are preserved alongside the implementation so that experiments remain reproducible and documentation can directly reference recorded metrics.",
            "Important directories include `IOT/` for the environmental branch, `crisis/` for the social-media branch, `XBD/` for the satellite branch, `fusion/` for the tri-modal serving and integration layer, `outputs-paper/` for evaluation artifacts, and `paper/` for earlier publication drafts."
        ]
    },
    {
        "title": "APPENDIX B KEY EXECUTION COMMANDS",
        "paragraphs": [
            "Representative training commands from the repository include `python IOT/train_iot.py` for the environmental metadata model and `python fusion/train_tri_fusion.py` for the tri-fusion layer. Notebook-based workflows are used for the crisis and satellite branches.",
            "The deployment-facing server can be started through the fusion subsystem, and the static dashboard assets under `fusion/static/` provide the browser-side interface for result visualization."
        ]
    }
]
