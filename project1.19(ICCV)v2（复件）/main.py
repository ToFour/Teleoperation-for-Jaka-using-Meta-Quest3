from RobotSys import*


#jakarobot 抓取 2025.12.28 

if __name__ == '__main__':
    robot_sys = RobotSys(ROBOT_IP)
    
    try:
        # 初始化系统
        robot_sys.initialize()
        print("系统初始化成功，开始执行任务...")
            
        robot_sys._move_to_safe_position()
        pick_pos=(725,408)
        result = robot_sys.pick(pick_pos) 
        print("抓取任务结果:", result["message"])
       
        place_pos=(484,477)
        robot_sys.place(place_pos)
        print("放置任务结果:", result["message"])




        robot_sys._move_to_safe_position()
            
     
      
    
    except Exception as e:
        print(f"主程序执行异常: {str(e)}")
    
    finally:
        # 关闭系统
        # robot_sys.shutdown()
        print(f"系统关闭")