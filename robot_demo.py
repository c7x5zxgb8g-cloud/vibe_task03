"""
微信机器人对接 Agent 回复系统 - 完整 Demo

本脚本模拟微信机器人与 Agent 服务端的完整交互流程：
  1. 机器人收到用户消息 → 上报给服务端
  2. 服务端后台自动分析意向 + 生成回复
  3. 机器人轮询获取待发送消息（含文件附件）
  4. 机器人发送后回调确认

启动方式：
  先启动服务端: uvicorn server:app --port 8000
  再运行本脚本: python robot_demo.py
"""

import time
import requests

# ============================================================
# 配置
# ============================================================
BASE_URL = "http://localhost:8000"


# ============================================================
# 工具函数
# ============================================================
def print_json(label: str, data):
    import json
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print()


# ============================================================
# 步骤 1: 机器人上报用户消息
# ============================================================
def step1_report_message():
    """
    机器人监听到用户消息后，调用此接口上报给服务端。
    服务端会自动进行意向分析 + 生成回复。
    """
    print("\n" + "#"*60)
    print("# 步骤 1: 机器人上报用户消息")
    print("#"*60)

    # ---- 场景 A: 群聊消息（高意向 → 通知飞书，等销售确认后才回复）----
    payload_group = {
        "sender_id": "wxid_demo_user_001",
        "sender_name": "张先生",
        "group_id": "group_12345",
        "group_name": "理博基金交流群",
        "content": "你们的量化选股产品最低多少钱可以投？收益怎么样？",
        "msg_type": "text",
        "msg_id": f"msg_group_{int(time.time())}",
        "timestamp": None  # 不填则服务端用当前时间
    }

    print_json("请求 POST /api/webhook/message (群聊-高意向)", payload_group)

    resp = requests.post(f"{BASE_URL}/api/webhook/message", json=payload_group)
    print_json("响应", resp.json())

    # ---- 场景 B: 私聊消息（低意向 → 自动从知识库回复，找不到则不回复）----
    payload_private = {
        "sender_id": "wxid_demo_user_002",
        "sender_name": "李女士",
        "group_id": "",          # 空 = 私聊
        "group_name": "",
        "content": "请问你们公司的创始人是什么背景？",
        "msg_type": "text",
        "msg_id": f"msg_private_{int(time.time())}",
    }

    print_json("请求 POST /api/webhook/message (私聊-客观问题)", payload_private)

    resp = requests.post(f"{BASE_URL}/api/webhook/message", json=payload_private)
    print_json("响应", resp.json())

    # ---- 场景 C: 私聊消息（知识库无法回答 → 不自动回复，等客服手动处理）----
    payload_private_unknown = {
        "sender_id": "wxid_demo_user_003",
        "sender_name": "王总",
        "group_id": "",
        "group_name": "",
        "content": "上次你们公司年会在哪里办的？",
        "msg_type": "text",
        "msg_id": f"msg_unknown_{int(time.time())}",
    }

    print_json("请求 POST /api/webhook/message (私聊-知识库外问题)", payload_private_unknown)

    resp = requests.post(f"{BASE_URL}/api/webhook/message", json=payload_private_unknown)
    print_json("响应", resp.json())


# ============================================================
# 步骤 2: 等待后台处理（意向分析 + 回复生成）
# ============================================================
def step2_wait_for_processing(wait_seconds=15):
    """
    服务端后台每 5 秒处理一批消息：
      - 意向分析（DeepSeek）
      - 高/中意向 → 预生成回复 + 推荐文件 → 飞书通知销售
      - 私聊低意向 → 检索知识库，有答案则自动排队回复
    """
    print("\n" + "#"*60)
    print(f"# 步骤 2: 等待后台处理 ({wait_seconds}s)")
    print("#"*60)
    print(f"  后台正在自动进行：")
    print(f"  - 意向分析（通过 DeepSeek）")
    print(f"  - 高/中意向客户 → 飞书通知销售确认")
    print(f"  - 私聊客观问题 → 知识库检索自动回复")
    print(f"  - 知识库无法回答 → 跳过，等待客服手动回复")

    for i in range(wait_seconds, 0, -1):
        print(f"\r  等待中... {i}s ", end="", flush=True)
        time.sleep(1)
    print("\r  处理完成!          ")


# ============================================================
# 步骤 3: （模拟销售）在管理后台确认跟进
# ============================================================
def step3_admin_confirm():
    """
    对于高/中意向客户，销售在飞书上看到通知后，
    在管理后台确认回复内容。

    实际流程：销售在飞书看到通知 → 点击链接到管理后台 → 确认跟进
    """
    print("\n" + "#"*60)
    print("# 步骤 3: 销售确认跟进（模拟管理后台操作）")
    print("#"*60)

    # 先登录
    login_resp = requests.post(f"{BASE_URL}/api/admin/login", json={"password": "admin123"})
    token = login_resp.json().get("token", "")
    headers = {"Authorization": f"Bearer {token}"}

    # 获取高意向线索
    leads_resp = requests.get(f"{BASE_URL}/api/admin/leads", headers=headers)
    leads = leads_resp.json().get("leads", [])

    if not leads:
        print("  暂无高意向线索")
        return

    # 取第一个线索，确认跟进
    lead = leads[0]
    customer_id = lead["customer_id"]
    print_json(f"线索 (customer_id={customer_id})", {
        "customer_name": lead.get("customer_name"),
        "intent_level": lead.get("intent_level"),
        "intent_score": lead.get("intent_score"),
        "message_content": lead.get("message_content"),
    })

    # 确认跟进 → 自动生成 AI 回复 + 推荐文件
    confirm_resp = requests.post(
        f"{BASE_URL}/api/admin/follow-up/{customer_id}/confirm",
        json={"admin_note": "客户意向较强，推荐量化选股产品资料"},
        headers=headers,
    )
    result = confirm_resp.json()

    print_json("确认跟进响应（含 AI 生成消息 + 推荐文件）", result)
    """
    期望响应格式:
    {
      "status": "ok",
      "follow_up_id": 1,
      "generated_message": "张先生您好！感谢您对我们量化选股产品的关注...",
      "attachment_files": [
          "理博基金产品介绍.pdf",
          "量化选股策略说明.docx"
      ]
    }
    """

    # 发送到队列
    if result.get("generated_message"):
        send_resp = requests.post(
            f"{BASE_URL}/api/admin/follow-up/send",
            json={
                "follow_up_id": result["follow_up_id"],
                "message": result["generated_message"],  # 可编辑后发送
            },
            headers=headers,
        )
        print_json("发送到队列响应", send_resp.json())


# ============================================================
# 步骤 4: 机器人轮询获取待发送消息（核心对接接口）
# ============================================================
def step4_poll_pending_messages():
    """
    机器人定时轮询此接口，获取所有待发送的消息。
    每条消息包含文本内容 + 文件附件下载链接。

    这是机器人最核心的对接接口。
    """
    print("\n" + "#"*60)
    print("# 步骤 4: 机器人轮询待发送消息")
    print("#"*60)

    resp = requests.get(f"{BASE_URL}/api/webhook/pending-messages")
    data = resp.json()

    print_json("GET /api/webhook/pending-messages 响应", data)

    """
    ============================================================
    响应格式说明:
    ============================================================
    {
      "messages": [
        {
          // ---- 基本信息 ----
          "follow_up_id": 1,              // 跟进记录ID（回调时需要）
          "target_user_id": "wxid_xxx",   // 目标用户的微信ID
          "group_name": "group_123", // 目标群ID（空=私聊）
          "customer_name": "张先生",       // 客户名称

          // ---- 消息内容 ----
          "content": "张先生您好！感谢您对我们量化选股产品的关注...",

          // ---- 附件文件（新增）----
          "attachment_files": [
            {
              "filename": "理博基金产品介绍.pdf",
              "download_url": "/api/materials/理博基金产品介绍.pdf"
            },
            {
              "filename": "量化选股策略说明.docx",
              "download_url": "/api/materials/量化选股策略说明.docx"
            }
          ]
        }
      ]
    }
    """

    # 模拟机器人处理每条消息
    for msg in data.get("messages", []):
        print(f"\n  📤 准备发送给: {msg['customer_name']} ({msg['target_user_id']})")
        print(f"  📝 消息内容: {msg['content'][:80]}...")

        # 处理附件文件
        attachments = msg.get("attachment_files", [])
        if attachments:
            print(f"  📎 附件数量: {len(attachments)}")
            for att in attachments:
                # 下载文件
                download_url = f"{BASE_URL}{att['download_url']}"
                print(f"     - {att['filename']}")
                print(f"       下载地址: {download_url}")

                # 机器人实际操作：下载文件 → 发送给用户
                # file_resp = requests.get(download_url)
                # save_path = f"/tmp/{att['filename']}"
                # with open(save_path, 'wb') as f:
                #     f.write(file_resp.content)
                # wechat_bot.send_file(msg['target_user_id'], save_path)
        else:
            print(f"  📎 无附件")

        # 发送完成后回调
        step5_callback_sent(msg["follow_up_id"], success=True)


# ============================================================
# 步骤 5: 机器人发送完成后回调
# ============================================================
def step5_callback_sent(follow_up_id: int, success: bool = True):
    """
    机器人发送消息后，回调此接口通知服务端发送结果。
    """
    print(f"\n  📞 回调: follow_up_id={follow_up_id}, success={success}")

    payload = {
        "follow_up_id": follow_up_id,
        "success": success,
        "error": "" if success else "发送失败: 用户不在线",
    }

    resp = requests.post(f"{BASE_URL}/api/webhook/message-sent", json=payload)
    print(f"  ✅ 回调结果: {resp.json()}")

    """
    请求格式:
    POST /api/webhook/message-sent
    {
      "follow_up_id": 1,       // 跟进记录ID
      "success": true,         // 是否发送成功
      "error": ""              // 失败原因（成功时为空）
    }

    响应格式:
    { "status": "ok" }
    """


# ============================================================
# 机器人轮询主循环示例
# ============================================================
def robot_polling_loop():
    """
    机器人实际运行时的轮询主循环模板。
    每 10 秒检查一次是否有待发送的消息。
    """
    print("\n" + "#"*60)
    print("# 机器人轮询主循环（示例模板）")
    print("#"*60)

    print("""
import time
import requests

BASE_URL = "http://your-server:8000"

def robot_main_loop():
    while True:
        try:
            # 1. 拉取待发送消息
            resp = requests.get(f"{BASE_URL}/api/webhook/pending-messages", timeout=10)
            data = resp.json()

            for msg in data.get("messages", []):
                follow_up_id = msg["follow_up_id"]
                target_user = msg["target_user_id"]
                target_group = msg.get("group_name", "")
                content = msg["content"]
                attachments = msg.get("attachment_files", [])

                try:
                    # 2. 发送文本消息
                    if target_group:
                        wechat_bot.send_group_message(target_group, content)
                    else:
                        wechat_bot.send_private_message(target_user, content)

                    # 3. 发送附件文件
                    for att in attachments:
                        file_url = f"{BASE_URL}{att['download_url']}"
                        file_data = requests.get(file_url).content
                        tmp_path = f"/tmp/{att['filename']}"
                        with open(tmp_path, 'wb') as f:
                            f.write(file_data)

                        if target_group:
                            wechat_bot.send_group_file(target_group, tmp_path)
                        else:
                            wechat_bot.send_private_file(target_user, tmp_path)

                    # 4. 回调成功
                    requests.post(f"{BASE_URL}/api/webhook/message-sent", json={
                        "follow_up_id": follow_up_id,
                        "success": True,
                        "error": ""
                    })

                except Exception as e:
                    # 5. 回调失败
                    requests.post(f"{BASE_URL}/api/webhook/message-sent", json={
                        "follow_up_id": follow_up_id,
                        "success": False,
                        "error": str(e)
                    })

        except Exception as e:
            print(f"轮询异常: {e}")

        time.sleep(10)  # 每 10 秒轮询一次
    """)


# ============================================================
# 接口速查表
# ============================================================
def print_api_reference():
    print("\n" + "#"*60)
    print("# Agent 对接机器人 - 接口速查表")
    print("#"*60)
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│                     机器人对接接口速查                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 上报消息（机器人 → 服务端）                                       │
│     POST /api/webhook/message                                        │
│     Body: {                                                          │
│       "sender_id": "wxid_xxx",      // 发送者微信ID                  │
│       "sender_name": "张先生",       // 发送者名称                    │
│       "group_id": "group_123",      // 群ID（空=私聊）               │
│       "group_name": "交流群",        // 群名称                       │
│       "content": "消息内容",         // 消息文本                      │
│       "msg_type": "text",           // 消息类型                      │
│       "msg_id": "unique_id",        // 消息去重ID                    │
│       "timestamp": null             // 时间戳（可选）                 │
│     }                                                                │
│                                                                      │
│  2. 拉取待发送消息（机器人轮询）                                      │
│     GET /api/webhook/pending-messages                                │
│     Response: {                                                      │
│       "messages": [{                                                 │
│         "follow_up_id": 1,                                           │
│         "target_user_id": "wxid_xxx",                                │
│         "group_name": "",                                       │
│         "content": "回复文本...",                                     │
│         "customer_name": "张先生",                                    │
│         "attachment_files": [{                                       │
│           "filename": "产品介绍.pdf",                                 │
│           "download_url": "/api/materials/产品介绍.pdf"               │
│         }]                                                           │
│       }]                                                             │
│     }                                                                │
│                                                                      │
│  3. 下载附件文件                                                     │
│     GET /api/materials/{filename}                                    │
│     Response: 文件二进制流                                            │
│                                                                      │
│  4. 发送结果回调                                                     │
│     POST /api/webhook/message-sent                                   │
│     Body: {                                                          │
│       "follow_up_id": 1,                                             │
│       "success": true,                                               │
│       "error": ""                                                    │
│     }                                                                │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                     回复生成逻辑                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  团队成员过滤:                                                       │
│    .env配置 WECHAT_TEAM_IDS 或在管理后台添加                          │
│    → 团队成员的消息自动忽略，不保存不回复                              │
│                                                                      │
│  普通消息回复（reply_type=auto，自动发送，不需审核）:                  │
│    群聊消息 → RAG知识库检索                                           │
│      → 知识库有答案 → 自动回复                                        │
│      → 知识库无答案 → 回复"不太确定，帮您问同事"                      │
│    私聊消息 → RAG知识库检索                                           │
│      → 知识库有答案 → 自动回复                                        │
│      → 知识库无答案 → 不回复，等客服手动处理                           │
│                                                                      │
│  跟单消息（reply_type=follow_up，需管理员审核才发送）:                 │
│    意向分析(DeepSeek) → 高/中意向 → 生成跟单消息+推荐文件             │
│    → 飞书通知销售 → 管理员审核通过 → 机器人发送                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
    """)


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  微信机器人对接 Agent 回复系统 - 完整 Demo")
    print("=" * 60)

    # 打印接口速查表
    print_api_reference()

    # 打印机器人轮询模板
    robot_polling_loop()

    # 以下为实际交互 demo（需要服务端运行）
    # 取消注释即可运行：

    # step1_report_message()       # 上报消息
    # step2_wait_for_processing()  # 等待后台处理
    # step3_admin_confirm()        # 销售确认跟进
    # step4_poll_pending_messages() # 机器人拉取消息
