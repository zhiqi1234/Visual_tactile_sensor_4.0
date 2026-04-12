import time
import os
os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
import topic
import message
import threading
# import python_message_bus

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# 1. 定义回调函数，处理接收到的 system_rtstate 消息
def on_system_rtstate_update(tt: topic.SystemRtState):
    safe_print("Received system rtstate update")
    parm_rt = message.SystemStateData()  # 注意：这里应该是 SystemStateData 而不是 message.SystemStateData
    message.display_rt(tt, parm_rt)

    safe_print("*********************** a.System State Update ********************************")

    safe_print(f"""
            1: header_timestamp: {parm_rt.header_timestamp}
            2: header_frame_id: {parm_rt.header_frame_id}
            3: system_running_state: {parm_rt.system_running_state}
            4: controller_name: {parm_rt.controller.controller_name}
            5: control_cycle: {parm_rt.controller.control_cycle}
            6: global_count: {parm_rt.controller.global_count}
            7: master_info: {parm_rt.controller.master_info}
            8: is_link_up: {parm_rt.controller.is_link_up}
    """)

    # 显示模型信息
    for model_idx, model_info in enumerate(parm_rt.models):
        safe_print(f"*********************** b.Model {model_idx} Information ********************************")
        safe_print(f"1: Model Name: {model_info.model_name}")
        safe_print(f"2: Model Type: {model_info.model_type}")
        
        # 显示当前点信息
        if model_idx < len(parm_rt.models_current_points):
            current_point = parm_rt.models_current_points[model_idx]
            safe_print(f"3: Point Name: {current_point.point_name}")
            safe_print(f"4: Tool Name: {current_point.tool_name}")
            safe_print(f"5: Wobj Name: {current_point.wobj_name}")
            safe_print(f"6: Tool Data: {current_point.tool_data}")
            safe_print(f"7: Wobj Data: {current_point.wobj_data}")
            safe_print(f"8: Robot Target: {current_point.robottarget}")
            safe_print(f"9: Joint Target: {current_point.jointtarget}")
        
        # 显示模型状态信息
        if model_idx < len(parm_rt.models_info):
            model_detail = parm_rt.models_info[model_idx]
            safe_print(f"10: Error Code: {model_detail.error_code}")
            safe_print(f"11: Error Message: {model_detail.error_msg}")
            safe_print(f"12: Model State: {model_detail.model_state}")
            safe_print(f"13: Model Time Rate: {model_detail.model_time_rate}")
            safe_print(f"14: Current Function Name: {model_detail.current_func_name}")
            safe_print(f"15: Current Function Info: {model_detail.current_func_info}")
            safe_print(f"16: Function Count: {model_detail.func_count}")

    # 计算每个模型的关节数量
    if parm_rt.models and parm_rt.models_joints:
        joints_per_model = len(parm_rt.models_joints) // len(parm_rt.models)
        
        # 显示关节信息（按模型分组）
        for model_idx in range(len(parm_rt.models)):
            safe_print(f"*********************** c.Model {model_idx} Joints Information ********************************")
            joint_start_idx = model_idx * joints_per_model
            joint_end_idx = joint_start_idx + joints_per_model
            
            for joint_idx, joint in enumerate(parm_rt.models_joints[joint_start_idx:joint_end_idx]):
                safe_print(f"Joint {joint_idx}:")
                safe_print(f"  Type: {joint.joint_type}")
                safe_print(f"  Position: {joint.position}")
                safe_print(f"  Torque: {joint.torque}")
                safe_print(f"  Is Enabled: {joint.is_enabled}")
                safe_print(f"  Mode: {joint.mode}")
                safe_print(f"  Error Code: {joint.error_code}")
                safe_print(f"  Digit Output: {joint.digit_output}")
                safe_print(f"  Digit Input: {joint.digit_input}")

    # 显示六维力传感器信息
    kk = 1
    safe_print("*********************** f.Force-Torque Sensor Information ********************************")
    for ftvalue in parm_rt.controller.ftvalues:
        safe_print(f"FT Sensor num: {kk}:  FX: {ftvalue.fx}")
        safe_print(f"FT Sensor num: {kk}:  FY: {ftvalue.fy}")
        safe_print(f"FT Sensor num: {kk}:  FZ: {ftvalue.fz}")
        safe_print(f"FT Sensor num: {kk}:  MX: {ftvalue.mx}")
        safe_print(f"FT Sensor num: {kk}:  MY: {ftvalue.my}")
        safe_print(f"FT Sensor num: {kk}:  MZ: {ftvalue.mz}")
        kk = kk + 1

# 定义回调函数，处理接收到的 system_nrtstate 消息
def on_system_nrtstate_update(tt: topic.SystemNrtState):
    safe_print("Received system nrtstate update")
    parm_nrt = message.SystemStateData()
    message.display_nrt(tt, parm_nrt)

    safe_print("*********************** A.System NRT State Update ********************************")

    safe_print(f"""
            1: header_timestamp: {parm_nrt.header_timestamp}
            2: header_frame_id: {parm_nrt.header_frame_id}
            3: system_running_state: {parm_nrt.system_running_state}
    """)
    
    # 显示从站信息
    safe_print("*********************** B.Controller Slaves Information ********************************")
    for slave_idx, slave in enumerate(parm_nrt.slaves):
        safe_print(f"Slave {slave_idx}:")
        safe_print(f"  Name: {slave.slave_name}")
        safe_print(f"  PHY ID: {slave.phy_id}")
        safe_print(f"  Alias: {slave.alias}")
        safe_print(f"  State: {slave.slave_state}")
        safe_print(f"  Is Online: {slave.is_online}")
        safe_print(f"  Is Virtual: {slave.is_virtual}")
        safe_print(f"  Is Error: {slave.is_error}")

    # 计算每个模型的数据量
    if parm_nrt.models:
        joints_per_model = len(parm_nrt.models_joints) // len(parm_nrt.models) if parm_nrt.models_joints else 0
        tools_per_model = len(parm_nrt.models_tools) // len(parm_nrt.models) if parm_nrt.models_tools else 0
        wobjs_per_model = len(parm_nrt.models_wobjs) // len(parm_nrt.models) if parm_nrt.models_wobjs else 0
        loads_per_model = len(parm_nrt.models_loads) // len(parm_nrt.models) if parm_nrt.models_loads else 0
        teach_points_per_model = len(parm_nrt.models_teach_points) // len(parm_nrt.models) if parm_nrt.models_teach_points else 0

    # 显示模型信息（按模型分组）
    for model_idx, model in enumerate(parm_nrt.models):
        safe_print(f"*********************** C.Model {model_idx} Information ********************************")
        safe_print(f"Model Name: {model.model_name}")
        safe_print(f"Model Type: {model.model_type}")
        safe_print(f"is_using_sp: {model.is_using_sp}")
        safe_print(f"is_collision_detection: {model.is_collision_detection}")


        # 显示关节信息
        if joints_per_model > 0:
            safe_print("  Joints:")
            joint_start_idx = model_idx * joints_per_model
            for joint_idx in range(joints_per_model):
                if joint_start_idx + joint_idx < len(parm_nrt.models_joints):
                    joint = parm_nrt.models_joints[joint_start_idx + joint_idx]
                    safe_print(f"    Joint {joint_idx}:")
                    safe_print(f"      Max Position: {joint.max_position}")
                    safe_print(f"      Min Position: {joint.min_position}")
                    safe_print(f"      Max Vel: {joint.max_vel}")
                    safe_print(f"      Min Vel: {joint.min_vel}")
                    safe_print(f"      Max Acc: {joint.max_acc}")
                    safe_print(f"      Min Acc: {joint.min_acc}")
                    safe_print(f"      Max Collision Torque: {joint.max_collision_torque}")

        # 显示工具信息
        if tools_per_model > 0:
            safe_print("  Tools:")
            tool_start_idx = model_idx * tools_per_model
            for tool_idx in range(tools_per_model):
                if tool_start_idx + tool_idx < len(parm_nrt.models_tools):
                    tool = parm_nrt.models_tools[tool_start_idx + tool_idx]
                    safe_print(f"    Tool {tool_idx}: {tool.tool_name}, Data: {tool.data}")

        # 显示工件坐标系信息
        if wobjs_per_model > 0:
            safe_print("  Work Objects:")
            wobj_start_idx = model_idx * wobjs_per_model
            for wobj_idx in range(wobjs_per_model):
                if wobj_start_idx + wobj_idx < len(parm_nrt.models_wobjs):
                    wobj = parm_nrt.models_wobjs[wobj_start_idx + wobj_idx]
                    safe_print(f"    Wobj {wobj_idx}: {wobj.wobj_name}, Data: {wobj.data}")

        # 显示负载信息
        if loads_per_model > 0:
            safe_print("  Loads:")
            load_start_idx = model_idx * loads_per_model
            for load_idx in range(loads_per_model):
                if load_start_idx + load_idx < len(parm_nrt.models_loads):
                    load = parm_nrt.models_loads[load_start_idx + load_idx]
                    safe_print(f"    Load {load_idx}: {load.load_name}, Data: {load.data}")

        # 显示示教点信息
        if teach_points_per_model > 0:
            safe_print("  Teach Points:")
            teach_point_start_idx = model_idx * teach_points_per_model
            for point_idx in range(teach_points_per_model):
                if teach_point_start_idx + point_idx < len(parm_nrt.models_teach_points):
                    point = parm_nrt.models_teach_points[teach_point_start_idx + point_idx]
                    safe_print(f"    Point {point_idx}: {point.point_name}")
                    safe_print(f"      Tool: {point.tool_name}, Wobj: {point.wobj_name}")
                    safe_print(f"      Tool Data: {point.tool_data}")
                    safe_print(f"      Wobj Data: {point.wobj_data}")
                    safe_print(f"      Robot Target: {point.robottarget}")
                    safe_print(f"      Joint Target: {point.jointtarget}")

    # 显示子系统信息
    safe_print("*********************** D.Subsystems Information ********************************")
    for subsystem_idx, subsystem in enumerate(parm_nrt.subsystems):
        safe_print(f"Subsystem {subsystem_idx}: {subsystem.subsystem_name}, ID: {subsystem.id}, State: {subsystem.state}")

    # 显示传感器信息
    safe_print("*********************** E.Sensors Information ********************************")
    for sensor_idx, sensor in enumerate(parm_nrt.sensors):
        safe_print(f"Sensor {sensor_idx}: {sensor.sensor_name}, ID: {sensor.id}, State: {sensor.state}")

    # 显示接口信息
    safe_print("*********************** F.Interfaces Information ********************************")
    for interface_idx, interface in enumerate(parm_nrt.interfaces):
        safe_print(f"Interface {interface_idx}: {interface.interface_name}, ID: {interface.id}, State: {interface.state}")

# system_rtstate中常用信息-快速使用
def on_system_basicstate_update(tt: topic.SystemRtState):
    safe_print("\n" * 1)
    safe_print("Received system basicstate update")
    parm_basic = message.SystemStateData()
    message.display_basic(tt, parm_basic)

    # 显示模型信息
    for model_idx, model_info in enumerate(parm_basic.models):
        safe_print(f"*********************** b.Model {model_idx} Information ********************************")
        safe_print(f"1: Model Name: {model_info.model_name}")
        safe_print(f"2: Model Type: {model_info.model_type}")
        
        # 显示当前点信息
        if model_idx < len(parm_basic.models_current_points):
            current_point = parm_basic.models_current_points[model_idx]
            safe_print(f"8: Robot Target: {current_point.robottarget}")
            safe_print(f"9: Joint Target: {current_point.jointtarget}")

    # 计算每个模型的关节数量
    if parm_basic.models and parm_basic.models_joints:
        joints_per_model = len(parm_basic.models_joints) // len(parm_basic.models)
        
        # 显示关节信息（按模型分组）
        for model_idx in range(len(parm_basic.models)):
            safe_print(f"*********************** c.Model {model_idx} Joints Information ********************************")
            joint_start_idx = model_idx * joints_per_model
            joint_end_idx = joint_start_idx + joints_per_model
            
            for joint_idx, joint in enumerate(parm_basic.models_joints[joint_start_idx:joint_end_idx]):
                safe_print(f"  Position: {joint.position}")
                safe_print(f"  Is Enabled: {joint.is_enabled}")


# 2. 配置节点选项
system_tpoic_options = topic.NodeOptions()
system_tpoic_options.node_name = 'test1'
system_tpoic_options.sub_url = 'tcp://192.168.50.1:19091'
safe_print("Subscribing1 to:", system_tpoic_options.sub_url)


# 3. 创建节点并启动
system_topic_node = topic.Node(system_tpoic_options)
if not system_topic_node.Start():
    safe_print("Failed to start node.")
else:
    safe_print("RTNode started successfully.")


# 4. 创建订阅
sub1 = system_topic_node.CreateSubscriptionRT("system_rtstate", on_system_rtstate_update)
sub2 = system_topic_node.CreateSubscriptionNRT("system_nrtstate", on_system_nrtstate_update)
sub3 = system_topic_node.CreateSubscriptionRT("system_rtstate", on_system_basicstate_update)


# 5. 保持程序运行，持续接收消息
try:
    while True:
        time.sleep(0.01)  # 防止主线程退出
except KeyboardInterrupt:
    safe_print("Shutting down...")
    system_topic_node.Shutdown()# 优雅关闭