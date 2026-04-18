import os
import tempfile
import requests
import streamlit as st
from typing import Optional, Tuple, Any, List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool

# =================================================================
# 1. NẠP BIẾN MÔI TRƯỜNG & CẤU HÌNH GIAO DIỆN
# =================================================================
load_dotenv()

OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
FAL_AI_API_KEY: Optional[str] = os.getenv("FAL_AI_API_KEY")

st.set_page_config(
    page_title="Sứ Giả Tịnh Độ - Bản Hoàn Thiện (Production)", 
    page_icon="🙏", 
    layout="wide"
)

if not OPENROUTER_API_KEY or not FAL_AI_API_KEY:
    st.error("❌ Lỗi Hệ Thống: Không tìm thấy API Key! Hãy kiểm tra lại file .env của bạn.")
    st.stop()

# =================================================================
# CẤU HÌNH LLM (OPENROUTER ROUTING)
# =================================================================
llm_gemma: LLM = LLM(
    model="openrouter/google/gemma-4-31b-it", 
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

llm_gemini: LLM = LLM(
    model="openrouter/google/gemini-3.1-pro-preview",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# =================================================================
# 2. CÔNG CỤ TẠO ẢNH (FLUX PRO)
# =================================================================
def generate_flux_image(prompt_text: str) -> Optional[str]:
    """Gọi API Fal.ai để tạo ảnh minh họa, xử lý nghiêm ngặt các ngoại lệ HTTP."""
    url: str = "https://fal.run/fal-ai/flux-pro"
    headers: dict = {
        "Authorization": f"Key {FAL_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload: dict = {
        "prompt": f"{prompt_text}, Buddhist Zen style, high-end photography, cinematic lighting, ethereal atmosphere",
        "image_size": "landscape_4_3"
    }
    
    try:
        res: requests.Response = requests.post(url, json=payload, headers=headers)
        
        if res.status_code == 404:
            st.error("❌ Lỗi 404: Endpoint API Fal.ai không tồn tại.")
            return None
        elif res.status_code in [401, 403]:
            st.warning("⚠️ Lỗi 403/401: Fal.ai từ chối truy cập. Kiểm tra lại số dư Credit.")
            return None
            
        res.raise_for_status() 
        return res.json()["images"][0]["url"]
        
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi API kết nối mạng: {str(e)}")
        return None
    except KeyError:
        st.error("Lỗi Parsing: Cấu trúc phản hồi từ máy chủ Fal.ai không hợp lệ.")
        return None

# =================================================================
# 3. THIẾT LẬP AGENTS & LUỒNG CÔNG VIỆC TỐI ƯU
# =================================================================
def start_crew_process(user_query: str, doc_path: Optional[str] = None, enable_image: bool = True) -> Tuple[str, Optional[str]]:
    """Thực thi AI Pipeline động. Có thể bypass Agent thiết kế để tiết kiệm Token."""
    
    file_tool: Optional[Any] = FileReadTool(file_path=doc_path) if doc_path else None
    tools: list = [file_tool] if file_tool else []
    
    # --- KHỞI TẠO CÁC AGENT CƠ BẢN (LUÔN CHẠY) ---
    writer: Agent = Agent(
        role='Sứ Giả Pháp Âm',
        goal='Tạo nội dung Facebook truyền cảm hứng từ tài liệu hoặc yêu cầu.',
        backstory="Chuyên gia Tịnh Độ Tông, ngôn từ an lạc, thấu cảm sâu sắc.",
        llm=llm_gemma,
        tools=tools,
        verbose=True
    )

    auditor: Agent = Agent(
        role='Nhạc Trưởng Kiểm Duyệt',
        goal='Đảm bảo đúng giáo lý, tối ưu Hook và Hashtag Facebook.',
        backstory="Chuyên gia Viral Marketing và gác cổng giáo lý Tịnh Độ.",
        llm=llm_gemini,
        verbose=True
    )

    task_write: Task = Task(
        description=f"Soạn bài viết Facebook sâu sắc dựa trên: {user_query}", 
        expected_output="Bản thảo nội dung bài đăng.", 
        agent=writer
    )
    
    task_audit: Task = Task(
        description="Kiểm duyệt lại giáo lý, chỉnh sửa định dạng, thêm Emoji và Hashtag phù hợp.", 
        expected_output="Bài đăng Facebook hoàn thiện cuối cùng.", 
        agent=auditor
    )

    # --- KHỞI TẠO MẢNG THỰC THI (DYNAMIC PIPELINE) ---
    active_agents: List[Agent] = [writer, auditor]
    active_tasks: List[Task] = [task_write, task_audit]

    # --- LOGIC ĐIỀU KIỆN (CHỈ CHẠY AGENT ẢNH NẾU ĐƯỢC BẬT) ---
    if enable_image:
        designer: Agent = Agent(
            role='Giám Đốc Nghệ Thuật',
            goal='Viết Prompt tiếng Anh nghệ thuật để vẽ ảnh minh họa. TUYỆT ĐỐI CHỈ XUẤT RA TIẾNG ANH.',
            backstory=(
                "Bậc thầy hình ảnh tâm linh. Bạn là một cỗ máy tạo Prompt. "
                "Bạn KHÔNG bao giờ chào hỏi, KHÔNG bao giờ giải thích. "
                "Đầu ra của bạn CHỈ LÀ một đoạn văn bản tiếng Anh mô tả hình ảnh."
            ),
            llm=llm_gemma,
            verbose=True
        )
        
        task_image: Task = Task(
            description=(
                "Dựa trên bài viết hoàn thiện vừa được kiểm duyệt, tạo 1 Prompt tiếng Anh chi tiết để vẽ ảnh.\n"
                "CẤM dùng các câu giao tiếp. Chỉ in ra nội dung mô tả bức ảnh."
            ), 
            expected_output="Một đoạn văn bản Tiếng Anh mô tả hình ảnh.", 
            agent=designer
        )
        # Chèn Agent thiết kế vào CUỐI pipeline để nhận text hoàn thiện nhất
        active_agents.append(designer)
        active_tasks.append(task_image)

    # Thực thi tiến trình
    crew: Crew = Crew(
        agents=active_agents, 
        tasks=active_tasks, 
        process=Process.sequential
    )
    
    crew.kickoff()
    
    # Bóc tách dữ liệu an toàn dựa trên độ dài của mảng tasks
    final_post: str = str(task_audit.output)
    image_prompt: Optional[str] = str(active_tasks[-1].output) if enable_image else None
    
    return final_post, image_prompt

# =================================================================
# 4. GIAO DIỆN STREAMLIT CHÍNH
# =================================================================
st.title("🙏 Sứ Giả Tịnh Độ - Hệ Thống Tự Động Hóa")

with st.sidebar:
    st.success("✅ Hệ thống đã Online (Connected)")
    
    # THÊM NÚT ĐIỀU KHIỂN LOGIC (STATE CONTROL)
    st.header("⚙️ Tùy chọn Xuất bản")
    use_image_generator = st.toggle("🎨 Tự động vẽ ảnh minh họa (Fal.ai)", value=True, help="Tắt để tiết kiệm Token/Chi phí nếu chỉ cần nội dung văn bản.")
    
    st.header("📁 Tài liệu nguồn")
    uploaded_file = st.file_uploader("Upload PDF/TXT để AI tự phân tích:", type=['pdf', 'txt'])

input_text: str = st.text_area(
    "Hôm nay bạn muốn lan tỏa điều gì?", 
    placeholder="Ví dụ: Nhìn mặt trời lặn thấy vô thường..."
)

if st.button("🚀 XUẤT BẢN NỘI DUNG"):
    if not input_text:
        st.warning("⚠️ Cảnh báo: Vui lòng nhập nội dung muốn lan tỏa trước khi xuất bản!")
    else:
        temp_file_path: Optional[str] = None
        
        try:
            # GIAI ĐOẠN 1: Tiến trình AI chạy ngầm (CrewAI)
            with st.spinner("Pháp âm đang được chép lại, vui lòng đợi trong giây lát..."):
                if uploaded_file is not None:
                    temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
                    temp_f.write(uploaded_file.getbuffer())
                    temp_f.close()
                    temp_file_path = temp_f.name
                
                # Truyền tham số cờ `use_image_generator` xuống core function
                content, prompt_for_img = start_crew_process(
                    user_query=input_text, 
                    doc_path=temp_file_path, 
                    enable_image=use_image_generator
                )
            
            # GIAI ĐOẠN 2: Render Text lên UI
            st.success("🎉 Hoàn tất quá trình tạo nội dung văn bản!")
            st.subheader("Bản thảo Bài Đăng")
            st.markdown(content)
            
            # GIAI ĐOẠN 3: Xử lý khối hình ảnh dựa trên cờ trạng thái
            if use_image_generator and prompt_for_img:
                with st.spinner("Đang phác họa không gian Tịnh Độ (Flux Pro)..."):
                    image_url: Optional[str] = generate_flux_image(prompt_text=prompt_for_img)
                
                if image_url:
                    st.subheader("Ảnh Minh Họa")
                    st.image(image_url, caption="Ảnh được kiến tạo bởi AI Fal.ai")
                else:
                    st.info("📌 Hệ thống không thể tải được ảnh. Vui lòng kiểm tra lại Dashboard của Fal.")
            else:
                st.info("📌 Chế độ vẽ ảnh đã được tắt. Hệ thống đã tiết kiệm chi phí cho bạn.")
                
        except Exception as e:
            st.error(f"💥 Lỗi nghiêm trọng trong quá trình vận hành: {str(e)}")
            
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)