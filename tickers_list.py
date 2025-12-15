"""
================================================================================
Script: tickers_list.py 
Created by: Anthony M. Schauer

--------------------------------------------------------------------------------
Overview:
Contains a Python list of all current S&P 500 tickers. This list is imported 
by data_pipeline.py so that the full S&P 500 data can be downloaded, processed, 
and normalized without manual editing of the pipeline code.
================================================================================
"""

sp500_tickers = [
    "NVDA", "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "AVGO", "META", "TSLA", 
    "BRK-B","WMT", "LLY", "JPM", "ORCL", "V", "XOM", "JNJ", "MA", "NFLX", 
    "PLTR","ABBV", "COST", "BAC", "AMD", "HD", "PG", "CSCO", "GE", "KO", "CVX",
    "UNH", "IBM", "MS", "MU", "WFC", "CAT", "GS", "AXP", "CRM", "APP",
    "MRK", "PM", "RTX", "TMUS", "MCD", "AMAT", "TMO", "ABT", "LRCX", "PEP",
    "ISRG", "C", "BX", "INTC", "DIS", "UBER", "QCOM", "LIN", "INTU", "NOW",
    "BLK", "T", "GEV", "VZ", "AMGN", "TJX", "APH", "SCHW", "ACN", "NEE",
    "BKNG", "TXN", "ANET", "KLAC", "DHR", "BA", "SPGI", "COF", "GILD", "PFE",
    "ADBE", "UNP", "BSX", "WELL", "LOW", "PANW", "ADI", "SYK", "ETN", "PGR",
    "CRWD", "MDT", "DE", "HOOD", "KKR", "HON", "PLD", "CB", "COP", "CEG",
    "VRTX", "HCA", "PH", "LMT", "ADP", "BMY", "NEM", "CVS", "MCK", "DASH",
    "MO", "CME", "CMCSA", "SO", "SBUX", "NKE", "DELL", "GD", "CDNS", "ICE",
    "DUK", "MMC", "TT", "SNPS", "MMM", "MCO", "WM", "APO", "AMT", "UPS",
    "BK", "USB", "ORLY", "SHW", "PNC", "NOC", "HWM", "GLW", "EMR", "MAR",
    "WMB", "ABNB", "COIN", "TDG", "AON", "CTAS", "ELV", "EQIX", "ECL", "MNST",
    "GM", "ITW", "REGN", "JCI", "CI", "MDLZ", "CMI", "WBD", "TEL", "PWR",
    "SPG", "RCL", "CSX", "COR", "NSC", "FDX", "RSG", "FCX", "ADSK", "TRV",
    "HLT", "FTNT", "AEP", "STX", "CL", "MSI", "AJG", "TFC", "KMI", "EOG",
    "WDAY", "AZO", "WDC", "ROST", "SRE", "NXPI", "MPC", "SLB", "PCAR", "PYPL",
    "DLR", "AFL", "IDXX", "VST", "PSX", "BDX", "DDOG", "VLO", "O", "APD",
    "ALL", "LHX", "F", "MET", "NDAQ", "ZTS", "URI", "EA", "D", "EW", "ROP", 
    "OKE", "CAH", "BKR", "PSA", "FAST", "MPWR", "TTWO", "CBRE", "GWW","FANG", 
    "AME", "ROK", "CMG", "CARR", "AMP", "LVS", "XEL", "CTVA", "EXC", "DAL", 
    "DHI", "AXON", "ETR", "TGT", "FICO", "AIG", "OXY", "PAYX", "MSCI",
    "KR", "PEG", "A", "TKO", "TRGP", "YUM", "KDP", "CCI", "PRU", "CTSH",
    "GRMN", "VMC", "VTR", "EBAY", "GEHC", "XYZ", "IQV", "EL", "CPRT", "MLM",
    "EQT", "HIG", "MCHP", "NUE", "RMD", "KEYS", "WAB", "HSY", "FISV", "STT",
    "SYY", "ED", "UAL", "WEC", "OTIS", "FIS", "KMB", "XYL", "CCL", "ACGL", 
    "PCG", "RJF", "NRG", "HPE", "EXPE", "LYV", "KVUE", "SNDK", "TER", "IR",
    "ODFL", "WTW", "FOXA", "HUM", "MTB", "FITB", "VRSK", "VICI", "CHTR",
    "IBKR", "SYF", "LEN", "K", "FOX", "KHC", "CSGP", "EXE", "ROL", "MTD",
    "EME", "EXR", "TSCO", "DG", "ADM", "FSLR", "DTE", "BRO", "ATO", "HBAN",
    "ULTA", "AEE", "CBOE", "BR", "DOV", "FE", "BIIB", "EFX", "STE", "WRB",
    "DXCM", "ES", "NTRS", "CINF", "PPL", "IRM", "AWK", "CNP", "STZ", "AVB",
    "VLTO", "GIS", "JBL", "STLD", "TDY", "TPR", "PHM", "DLTR", "HAL", "LDOS",
    "CFG", "HPQ", "DVN", "EQR", "HUBB", "RF", "WAT", "NTAP", "TROW", "VRSN",
    "PPG", "ON", "KEY", "EIX", "RL", "LULU", "CMS", "LH", "WSM", "CPAY",
    "L", "PODD", "NVR", "SMCI", "DRI", "PTC", "TPL", "CTRA", "SBAC", "IP",
    "DGX", "CHD", "LUV", "NI", "EXPD", "TSN", "TYL", "TRMB", "PFG", "WST",
    "TTD", "CDW", "INCY", "AMCR", "CNC", "SW", "GPN", "ZBH", "CHRW", "SNA",
    "JBHT", "BG", "Q", "LII", "GPC", "PKG", "GDDY", "DD", "FTV", "MKC", "EVRG", 
    "ESS", "PNR", "GEN", "APTV", "HOLX", "LNT", "IT", "IFF", "DOW", "J", 
    "INVH", "WY", "MAA", "BBY", "PSKY", "ALB", "COO", "NWS", "FFIV", "TXT", 
    "DECK", "ERIE", "NWSA", "DPZ", "OMC", "UHS", "LYB", "SOLV","BF-B", "ALLE", 
    "KIM", "ZBRA", "AVY", "NDSN", "JKHY", "HRL", "EG", "IEX", "VTRS", "REG", 
    "WYNN", "MAS", "UDR", "BALL", "AKAM", "HII", "CLX", "BXP", "BEN", "HST", 
    "CF", "DOC", "IVZ", "BLDR", "ALGN", "EPAM", "RVTY", "HAS", "AIZ", "SWK", 
    "DAY", "MRNA", "CPT", "FDS", "GL", "SJM", "PNW", "SWKS", "AES", "MGM", 
    "GNRC", "BAX", "APA", "TECH", "CRL", "AOS", "TAP", "PAYC", "POOL", "HSIC", 
    "NCLH", "FRT", "CPB", "DVA", "CAG", "LW", "MTCH", "MOH", "ARE", "SOLS"
]

