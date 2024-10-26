PID=$(ps aux | grep '[a]pp.py' | awk '{print $2}')

if [ -z "$PID" ]; then
  echo "smart_city is currently NOT running"
else
  echo "stopping smart_city (PID: $PID)..."
  kill -9 $PID
  echo "smart_city stopped。"
fi
./smart_city/bin/pip3  install -r requirements.txt 

nohup ./smart_city/bin/python3 app.py --host=0.0.0.0 --port=5000 > app.log 2>&1 &
