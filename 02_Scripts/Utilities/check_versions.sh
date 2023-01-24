echo "Checking versions of python packages"

echo "Pandas" 
pip show pandas | grep Version

echo "numpy"
pip show numpy | grep Version

echo "scipy"
pip show scipy | grep Version

echo "matplotlib"
pip show matplotlib | grep Version

echo "plotly"
pip show plotly | grep Version

echo "GitPython"
pip show GitPython | grep Version

echo "pybaselines"
pip show pybaselines | grep Version