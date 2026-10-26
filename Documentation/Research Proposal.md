

Manchester Metropolitan University 



<!-- Start of picture text -->
— baa i |e SA é| | — =<br>rite half)? 10° ee | |<br>f bm ies & | a: bc 4 m4 a a oa Ee ote -- ay! la b a 1 1.<br>q — 5 a | LB ats) a | 1 : 7 eo eee. . i i<br>ia || —-i e >) ‘a /— > ae ig te, eel sd Wee ie 1G Fy ial TT wT ei : - —_— oe 4<br>DSS EZ<br><!-- End of picture text -->

@ 



<!-- Start of picture text -->
@<br><!-- End of picture text -->

# **Literature Review** 

**Research Gap** Most studies focus only on object detection. Limited integration of OCR and quality analytics. 

Few systems combine vision, weight, and cloud logging. 

The literature shows that many researchers have successfully applied YOLOv5 for industrial inspection. However, most existing systems focus only on defect detection. Our work integrates object detection, OCR, weight inspection, and cloud analytics into one complete framework. 

# **Existing Research** 



<!-- Start of picture text -->
Author<br><!-- End of picture text -->

**Author Contribution** YOLOv5 for Elbaghdadi et al. industrial defect (2026) detection Enhanced YOLOv5 Fang et al. (2024) for manufacturing inspection Deep learning for Yang et al. (2021) weld defect detection JaidedAI EasyOCR OCR for text extraction 



<!-- Start of picture text -->
Pe er<br>/ CONVEYOR \<br>I i<br>SIMPLE jeees<br>FLOWCHART :<br>VISION INSPECTION<br>AND EASYOCR<br>WEIGHT AND SHAPE<br>INSPECTION<br>REJECT PASS<br>REJECTED Gesssssasomerezaaa: PASS? ste ROBOT PLACE<br>REJECTION<br>LOGGER<br>CLOUD DATA STORAGE<br>R&D ANALYTIC DASHBOARD<br><!-- End of picture text -->

@ 



<!-- Start of picture text -->
@<br>e<br><!-- End of picture text -->

@ 

a 

@ 

@ 



<!-- Start of picture text -->
Image acquisition<br>OpenCV captures live feed<br>Image preprocessing<br>Resize, normalize, denoise<br>YOLOv5 object detection<br>Detects cans & labels in real-time<br>Yes No<br>Flag X<br><!-- End of picture text -->

Stan 4 

Input image Tin can photo from camera 

Step 2 

Step 3 

| 

Image preprocessing Resize, normalize, augment (OpenCV) | 

YOLOvs backbone CSPDarknet53 — deep feature extraction 



<!-- Start of picture text -->
@<br><!-- End of picture text -->

Step 4 

Neck — multi-scale feature fusion PANet merges small and large object features 



<!-- Start of picture text -->
@<br><!-- End of picture text -->

Step 5 

Step 7 

Step 8 

Step 9 

Step<sup>10</sup> 

( 

Detection head Predict boxes across 3 scales 

Bounding boxes + confidence + class Box coords, loU confidence score, defect class label (dent, rust, scratch...) } 

Non-maximum suppression (NMS) Remove duplicate overlapping predictions 

EasyOCR label verification Read and validate product text on can 

Cloud analytics Real-time monitoring, Industry 4.0 logging 

Final defect detection result Pass / Fail verdict + defect map output 

e 



<!-- Start of picture text -->
@<br><!-- End of picture text -->



<!-- Start of picture text -->
@<br><!-- End of picture text -->



<!-- Start of picture text -->
e<br><!-- End of picture text -->



<!-- Start of picture text -->
EasyOCR scan<br>Extract serial number from label<br>Read mfg & expiry dates<br>Multilingual OCR recognition<br>Shape analysis: (OpenCV)<br>Contour & dimension checks<br>Weight verification<br>Load cell reading vs. threshold<br>No Yes<br>Reject x All checks pass?> Accept Y<br><!-- End of picture text -->

° e 



<!-- Start of picture text -->
Product rejected<br>Trigger cloud logging pipeline<br><!-- End of picture text -->

e 



<!-- Start of picture text -->
Log rejection details<br>Reason, timestamp, product ID<br><!-- End of picture text -->

e 

Save images & sensor data Cloud blob + database storage 

Generate rejection reports Automated PDF / export 

Dashboard visualisation Trend charts, defect rates, KPls 

wee ee eeep> 

Predictive maintenance 

Inspection Report.csv 

: x 

1 entry 

e 

e 

e 

e 

e 



<!-- Start of picture text -->
ia<br><!-- End of picture text -->

Weight(g) Shape OCR Defect Decision 496 Normal Ime '#' Lemoni,Limca; Scratch REJECT 

### Show|10 ~|per page 

## eee Tin Can Inspection 

#### OCR Results 

["Ime *#° Lemoni”, ‘Limca;'] Shape : Normal Measured Weight : 496 g Weight : PASS Detected Defect : Scratch Final Decision : REJECT Manufacturing Analytics Dashboard 

Total Products : 1 Accepted : @ Rejected : 1 Rejection Rate : 100.0 % 

##### Inspection Report Saved Successfully 



<!-- Start of picture text -->
‘<br><!-- End of picture text -->



<!-- Start of picture text -->
|<br>une Th LEMON<br>, |<br>Lyne<br>AWA a<br>WNSe<br>Tans .<br><!-- End of picture text -->

Defect Distribution Defect Scratch 1 Name: count, dtype: int64 

Average Weight 496.0 g 

**Conclusion** Developed an AI-based inspection framework for tin can manufacturing. YOLOv5 enables fast and accurate defect detection. EasyOCR automatically verifies product label information. OpenCV performs image preprocessing and shape analysis. Cloud analytics supports real-time monitoring and Industry 4.0 manufacturing. The proposed system reduces manual inspection effort and improves quality control. 

**_Future Work_** 

**_References_** 

Potential extensions after successful completion of this project: Upgrade the inspection model from YOLOv5 to YOLOv8/YOLOv11 for improved detection accuracy. 

Deploy the solution on Edge AI devices for faster real-time processing. 

Integrate a robotic arm for automatic removal of defective products. 

Apply predictive quality maintenance using historical inspection data. 

Extend the system to inspect multiple product types within the production line. 

Elbaghdadi, A., et al. (2026). 

YOLOv5: An Improved Algorithm for Real-Time Detection of Industrial Defective Pieces. 

Fang, Y., et al. (2024). 

Enhanced YOLOv5s-Based Algorithm for Industrial Part Detection. 

Yang, D., et al. (2021). 

Deep Learning-Based Steel Pipe Weld Defect Detection Using YOLOv5. 

JaidedAI. 

EasyOCR Documentation. https://github.com/JaidedAI/EasyOCR 

Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools. 

