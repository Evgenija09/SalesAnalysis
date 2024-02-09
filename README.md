## 1. Big Data Analysis

### 1.1 About the dataset and my objectives

The dataset provided captures extensive information pertaining to sales orders, likely within the domain of e-commerce or retail, with a specific focus on a downhill bike and accessories shop. A preliminary exploration of the data revealed that numeric columns such as OrderQuantity, UnitPrice, SalesAmount, and Freight display considerable variability, meaning these parameters have a diverse range. Also, there are outliers, especially in maximum values, which prompts to data distribution meaning there needs to be deeper analysis (shown in Figure 1).

For a more detailed understanding, key statistics provided insights into the distribution and range of the data. For instance, OrderQuantity has a mean of 6.05 with a high standard deviation of 60.99, indicating significant variation. UnitPrice, with an average of $467.47 and a standard deviation of $911.74, suggests a wide range in product pricing. SalesAmount, averaging $13,365.98 with a substantial standard deviation of $28,350.56, hints at considerable variation in transaction values. Freight, averaging $12.15 with a high standard deviation of $225.64, highlights variability in shipping costs.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/1a82694e-8e5f-4f17-a74f-30138d7c5d23)

*Figure 1: Boxplot of numeric values*

The preliminary analysis prompted defining objectives for future analysis. Several possibilities emerged, including geographical variations, financial metrics, promotional impacts, sales performance, and customer behavior. Having these objectives into focus, the forthcoming section will address the preprocessing of the dataset with the goal to render the data more manageable, optimize it for analysis, and prepare it for machine learning.

### 1.2 Big data preprocessing

In the preprocessing phase, the focus was refining the dataset to align with the mentioned objectives. That being said, the first step was to remove the tables that were irrelevant to my objectives, leaving only 11 tables. The next phase involved addressing missing values for which I created a function, showcased in Figure 2. This function systematically examines each table, presenting a breakdown of missing values for individual columns. There are also statistics, including absolute counts and percentages relative to the total rows, which offered a clear overview of the data gaps. This aided in identifying columns with over 50% missing values, which were removed to avoid noise. Additionally, I made notes regarding columns such as ListPrice, Size, Weight, ProductLine, DealerPrice, Class, and Style, which exhibited missing values (20%-30%) in the initial rows. The decision not to delete those rows stemmed from the possibility that these rows contain valuable data for other columns that will be needed in future analyses.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/a51d4064-2ff1-4c74-9ce1-cb3b35d9104d)

*Figure 2: Function to find missing values*

The subsequent preprocessing step involved refining data types using the code in Figure 3. The function checked and adjusted column data types—converting integers to IntegerType, doubles to DoubleType, and dates to DateType - enhancing uniformity.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/54908fde-7836-4b86-aea3-371ba87ea24b)

*Figure 3: Function to check data types*

Additionally, in Figure 4 you can see an example where the 'DimProduct' table underwent date conversion using PySpark's to_date function. 'EndDate' and 'StartDate' columns were transformed for standardized date representation in the dataset.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/2a7dba1e-5805-4524-8876-3237e66850c1)

*Figure 4: Date type conversion*

Lastly, categorical columns with limited distinct values (less than 10) were identified and transformed accordingly. The code snippet in Figure 5 is an example for the 'DimCustomer's 'CommuteDistance' column.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/dea07250-7fca-4dc0-8284-1071f8d1509a)

*Figure 5: Categorical transformations*

### 1.3 Exploratory data analysis (EDA)

In this section, analysis and visualizations are applied to the preprocessed data in order to gain deeper understanding into its underlying patterns and characteristics. Firstly, we explore the sales trajectory over time. The line plot presented in Figure 6, is a visual depiction of sales fluctuations, offering insights into patterns and trends. Notably, between 2011 and 2012, sales showcased peaks at 3500 and lows at 500, reflecting robust performance. However, from 2012 to 2013, a decline ensued, reaching highs of 2400 and lows of 700. January 2013 marked a significant drop to 10-100 in sales, indicating challenges. Post-March 2013, sales consistently hovered around 400, signaling a sustained period of lower performance.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/297b7501-849e-41c9-9d5f-a3856c7ad5cf)

*Figure 6: Sales Trends Over Time Lineplot*

Moving on, in order to identify and prioritize top-performing product categories for strategic decision-making I used temporary views and SQL queries (Figure 7), which gave insights product categories-sales performance. The data extracted suggested that the "Road Bike" subcategory had consistent sales of 3578.27 throughout 2010 and 2011, with a focus on the first month of each year. This pattern indicates the most active periods for this product, highlighting a potential trend in consumer behavior during the early months.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/0066da66-7f71-402b-9af4-3cbb1d944c60)

*Figure 7: Top performing product categories*

The next step, customer segmentation analysis, achieved through SQL queries on 'DimCustomer' and 'FactInternetSales,' discerned distinct purchasing behaviors. Visible in Figure 8, customers like Nichole Nara and Kaitlyn Henderson exhibited high total sales, providing valuable insights for personalized marketing strategies.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/b41d7f38-c4d9-4862-9f0f-d56a20276a8a)

*Figure 8: Customer Segmentation Analysis*

With the boxplot in Figure 9, the aim was to gain insights into geographical sales patterns. The results show the city Milsons Point in New South Wales, Australia, has the highest total sales. These results can be used for targeted marketing and resource allocation.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/572a1351-2013-4456-82f1-252b2ee9cc23)

*Figure 9: Boxplot for geographical sales patterns*

The last part with the code presented in Figure 10 focused on customer loyalty, identifying frequent purchasers. Loyal customers, like Ashley Henderson with 68 purchases (Figure 11), are crucial for targeted marketing and fostering brand loyalty.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/43637f0a-79ec-45d2-9ad9-63b2c32cfeb0)

*Figure 10: Find loyal customers*

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/f852bb49-64ea-444e-802c-4d7e47af13b2)

*Figure 11: Tables displaying loyal customers*

### 1.4 Machine learning algorithms

In this section, the primary objective was to gain insights into sales, customers, and products. The initial analysis applied linear regression to forecast SalesAmount. The model demonstrated very good performance with a Mean Squared Error of 6660.46 and a high R-squared value of 0.9947, indicating a strong fit. Key features influencing predictions included StandardCost, Weight, and DaysToManufacture. Negative coefficients in feature importances (ColorIndex, SizeIndex, DaysToManufacture) suggest an increase in these factors is associated with a decrease in predicted SalesAmount, indicating a negative impact on sales. Conversely, positive coefficients (StandardCost, Weight) suggest an increase in these factors correlates with an increase in predicted SalesAmount, indicating a positive impact on sales. Visualizing the model's performance revealed an upward-sloping line visible in Figure 14, indicating accurate predictions with positive correlations though some deviations from a perfectly straight line are visible.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/a1ffaf3a-c93f-4e66-ac34-43694f2c08a5)

*Figure 12: Linear regression analysis*

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/755b0269-1b7c-415f-826b-e05279455ec5)

*Figure 13: Feature importances*

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/1bd745b5-6ddf-464e-9fe8-d3c065a289cf)

*Figure 14: Visualization of linear regression results*

The second part was applying a K-means clustering with the goal to identify patterns and groupings within promotions based on their impact on sales. The objective was to efficiently cluster promotions by leveraging features from both 'DimPromotion' and 'FactInternetSales' tables. This identified three distinct promotion clusters with silhouette score (0.90) and low inertia (101.30). The results from Figure 15 show that Cluster 1 exhibits lower sales, tax, and freight, suggesting less impactful promotions. Cluster 2 showcases significantly higher sales, tax, and freight, indicating potentially successful strategies. Cluster 3 falls between the two in terms of sales-related metrics. The PCA plot in Figure 16 shows coherent clusters, with the yellow group displaying lower cohesion and also one present outlier. 

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/2478af36-5fe6-455a-b136-2d001be2cf4b)

*Figure 15: Results from K-mean clustering*

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/b1285c83-73f7-4862-bd62-b2c86d2785a6)

*Figure 16: PCA K-means clustering*

Lastly, I conducted a product classification analysis using RandomForestClassifier using the code in Figure 17. Here, the model achieved an overall accuracy of 83%. Notably, it exhibited high precision, recall, and F1-scores for various product categories, such as 'Bike Wash,' 'Classic Vest,' and 'Racing Socks' as it is visible in Figure 18.  The confusion matrix in Figure 19 reveals strong performance in accurately classifying popular products, while challenges arose in distinguishing less common items like 'All-Purpose Bike Stand' and 'Touring-3000,' leading to lower precision and recall. This suggests the model's proficiency in recognizing popular products but potential limitations when classifying less prevalent ones.

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/c308607d-b22c-43ee-9f6b-25658926def2)

*Figure 17: RandomForestClasifier for geographical sales analysis*

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/27cc2de4-dff5-418f-aace-a689940cca55)

*Figure 18: Results from RandomForestClassifier*

![image](https://github.com/Evgenija09/SalesAnalysis/assets/45256672/b74e06df-81ff-40c9-9f56-3ffe95e0327e)

*Figure 19: Confusion matrix for RandomForestClassifier results*


