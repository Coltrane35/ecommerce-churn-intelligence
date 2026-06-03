-- Customer-level transaction features for churn / CLV modeling

SELECT
    CustomerID,
    COUNT(DISTINCT InvoiceNo) AS frequency_orders,
    SUM(Quantity * UnitPrice) AS monetary_total,
    AVG(Quantity * UnitPrice) AS monetary_mean,
    MAX(InvoiceDate) AS last_purchase_date
FROM transactions
WHERE
    CustomerID IS NOT NULL
    AND Quantity > 0
    AND UnitPrice > 0
GROUP BY CustomerID;