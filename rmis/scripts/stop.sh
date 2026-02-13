user=$(whoami)

# kill main process
ps -u $user -f | grep "python -m rmis.scripts.reg_all" | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
ps -u $user -f | grep "python -m rmis.scripts.multi_split_all" | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}

# kill sub process
ps -u $user -f | grep "python -m runner.im.im" | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
ps -u $user -f | grep "python -m runner.cls.cls_reg_nrun" | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
ps -u $user -f | grep "python -m runner.cls.multi_split_reg_nrun" | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
