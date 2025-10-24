# Dataset Features

## Overview
- Grain: One row per `period_id` × `customer_salesarea_sk` (POC) × `product_salesarea_sk`.
- Domain coverage: POC Sell‑In, POC Sell‑Out, Trade Inventory, Market Share, Promotions.
- Geography: Mexico (MX) with region/city and Nielsen market IDs.
- Time coverage: 18 months total — explicitly 2024‑01 .. 2024‑09 and 2025‑01 .. 2025‑09.

- What this data represents
  - A synthetic beverages dataset designed for demoing commercial analytics at Point‑of‑Customer (store/DC) level.
  - Basic items are beverage SKUs grouped into three categories: `CSD` (carbonated soft drinks), `JUICE`, and `WATER`.
  - Each product has simple attributes: `product_flavour` (e.g., Cola, Lime, Orange…), `packaging_code` (PET, CAN, GLS), and a price tier (via `retail_price_lcl`/`net_price_lcl`).
  - Points of customer (POCs) are represented by `customer_salesarea_sk` and are observed across regions/cities and distribution channels.

## Columns
- Time and geography
  - `period_id` (yyyymm)
  - `period_end_date` (YYYY‑MM‑DD)
  - `calendar_name` (e.g., MX_NATURAL)
  - `country_code` (e.g., MX)
  - `region_code` (e.g., SLP, JAL, TMS)
  - `city_name` (e.g., SAN LUIS POTOSI, PUERTO VALLARTA, ALTAMIRA)
  - `nielsen_id` (e.g., I2, I3)
  - `latitude`, `longitude`

- Customer and product
  - `sales_org` (e.g., MX02)
  - `distribution_channel` (20, 45)
  - `division_code` (1)
  - `customer_salesarea_sk` (stable numeric surrogate of POC)
  - `product_salesarea_sk` (stable numeric surrogate of product)
  - `location_type` (STORE | DC)
  - `category` (CSD | JUICE | WATER)
  - `product_flavour` (e.g., Cola, Lime, Orange, …)
  - `packaging_code` (PET, CAN, GLS)

- Sell‑In (POC)
  - `si_orders_cases` — orders in cases
  - `si_orders_value_lcl` — orders value (local currency)
  - `si_invoices_cases` — billed cases
  - `si_invoices_value_lcl` — billed value (local)
  - `si_total_virtual_cases` — total virtual cases
  - `si_total_virtual_value_lcl` — total virtual value (local)
  - `si_net_revenue_value_lcl` — net revenue (local)
  - `si_returns_value_lcl` — returns value (local)
  - `si_gross_profit_value_lcl` — gross profit value (local)
  - `si_returns_rate` — returns rate (0.0000–1.0000)
  - `si_asp_value_per_case` — average selling price per case (local)

- Sell‑Out (POC)
  - `so_units` — units sold (aligned to cases for simplicity)
  - `so_value_lcl` — sell‑out value (local)
  - `so_returns_units` — returned units (derived from price and returns value)
  - `so_returns_value_lcl` — returned value (local)
  - `retail_price_lcl` — list price (local)
  - `net_price_lcl` — realized net price after discount (local)

- Trade Inventory
  - `opening_stock_units`
  - `receipts_units` (≈ `si_invoices_cases`)
  - `shipments_units` (≈ `so_units`)
  - `adjustments_units` (small ± corrections)
  - `closing_stock_units` = opening + receipts − shipments + adjustments
  - `days_of_supply` — closing ÷ max(daily sell‑out, 1)

- Promotions
  - `promo_id` (PR‑<period>‑<customer>‑<product> when active, else empty)
  - `promo_active_flag` (0/1)
  - `promo_type` (TPR | Feature | Display | None)
  - `promo_mechanic` (Price Cut | Bundle | BOGO | None)
  - `discount_pct` (integer percent)
  - `feature_flag`, `display_flag`, `media_support` (0/1)

- Market / Share
  - `market_units` — inferred market size (units)
  - `market_value_lcl` — inferred market value (local)
  - `share_units_pct` — internal units / market units (0.00–1.00)
  - `share_value_pct` — internal value / market value (0.00–1.00)
  - `share_rank` — rank 1..5

## KPIs (derived or directly usable)
- Sell‑in volume/value: use `si_*` columns
- Net revenue: `si_net_revenue_value_lcl`
- Returns and returns rate: `si_returns_value_lcl`, `si_returns_rate`
- Gross profit: `si_gross_profit_value_lcl`
- ASP: `si_asp_value_per_case`
- Sell‑out volume/value: `so_units`, `so_value_lcl`
- Inventory coverage: `days_of_supply`, plus stock movement columns
- Promotions: activity, mechanics, depth (`promo_*`, `discount_pct`)
- Market share: `share_units_pct`, `share_value_pct`, `share_rank`

## Time Coverage
- Explicit months included:
  - 2024: 202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409
  - 2025: 202501, 202502, 202503, 202504, 202505, 202506, 202507, 202508, 202509
- `period_end_date` corresponds to the last day of each month.
- YoY analysis: compare 2024‑MM vs 2025‑MM for Jan..Sep.

## Generation Logic (assumptions)
- Seasonal factors per category (applied to base volume):
  - CSD: low in Jan/Feb, peak Jun–Aug, easing from Sep
  - JUICE: modest lift in summer
  - WATER: strong spring/summer peak May–Aug

- Channel/region multipliers:
  - Channel: `distribution_channel=45` gets +10% volume and higher promo propensity; deeper discounts.
  - Region boosts: WATER higher in JAL; CSD slightly higher in SLP; JUICE slightly lower in TMS.

- 2025 inflation and unit growth:
  - Prices: +6% list price inflation in 2025; `retail_price_lcl` reflects inflation and decimals.
  - Volume: +5% uplift to 2025 base units after seasonal and channel/region effects.

- Promotions (probability and depth):
  - Base activation: 25% in non‑summer months, 45% in May–Aug.
  - +10% activation if channel=45; +5% if category=CSD.
  - Discount depth (ranges): deeper for channel=45 and CSD; otherwise moderate.
  - `promo_type` ∈ {TPR, Feature, Display} with simple random assignment when active.

- Sell‑in vs sell‑out:
  - `si_invoices_cases` ≈ `si_orders_cases` × (0.92 .. 1.00).
  - `si_total_virtual_cases` ≈ `si_invoices_cases` ±10%.
  - `si_net_revenue_value_lcl` ≈ `si_invoices_value_lcl` × 0.94.
  - `si_returns_value_lcl` = 0–3% of invoice value; `si_returns_rate` derived.
  - `si_gross_profit_value_lcl` ≈ 35% of net revenue.
  - `so_units` ≈ invoices with +10% promo lift and ±8% noise.

- Inventory modeling:
  - `receipts_units` = `si_invoices_cases`.
  - `shipments_units` = `so_units`.
  - `closing_stock_units` computed from opening, receipts, shipments, and small adjustments.
  - `days_of_supply` = closing / max(daily sell‑out, 1).

- Market totals and share:
  - `market_units` and `market_value_lcl` inferred from internal results and a target share range per category.
  - `share_units_pct`, `share_value_pct` reported as 0.00–1.00 proportions; `share_rank` 1..5.

## Topic Mapping
- POC Sell‑In
  - Use `si_orders_*`, `si_invoices_*`, `si_total_virtual_*`, `si_net_revenue_value_lcl`, `si_returns_*`, `si_gross_profit_value_lcl`, `si_asp_value_per_case`.
  - Segment by POC/product/time, with `distribution_channel`, `region_code`, `city_name` for cuts.

- POC Sell‑Out
  - Use `so_units`, `so_value_lcl`, `so_returns_*`, `retail_price_lcl`, `net_price_lcl`.
  - Join context via the same POC/product/time grain to compare against sell‑in.

- Trade Inventory
  - Use `opening_stock_units`, `receipts_units`, `shipments_units`, `adjustments_units`, `closing_stock_units`, `days_of_supply`.
  - Compute coverage trends and OOS risk by POC/product/time.

- Market Share
  - Use `market_units`, `market_value_lcl`, `share_units_pct`, `share_value_pct`, `share_rank`.
  - Segment by `category`, `region_code`, `nielsen_id`, `period_id`.

- Promotions
  - Use `promo_active_flag`, `promo_id`, `promo_type`, `promo_mechanic`, `discount_pct`, `feature_flag`, `display_flag`, `media_support`.
  - Analyze promo incidence, depth, and their uplift on `so_units`/`so_value_lcl` and price realization (`net_price_lcl`).

## Notes
- Currency fields (`*_value_lcl`, prices) are formatted to two decimals; rates use four decimals where applicable.
- Units in `so_units` align to case‑like units for simplicity across topics.
