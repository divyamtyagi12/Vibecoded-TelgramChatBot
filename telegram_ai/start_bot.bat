@echo off
title Telegram AI Bot
cd /d "C:\Users\divyam\Desktop\telegram_ai"

:LOOP
echo [%date% %time%] Starting bot...
"C:\Users\divyam\AppData\Local\Programs\Python\Python313\python.exe" -m telegram_ai >> bot.current.stdout.log 2>> bot.current.stderr.log
echo [%date% %time%] Bot stopped. Restarting in 5 seconds...
timeout /t 5 /nobreak
goto LOOP
