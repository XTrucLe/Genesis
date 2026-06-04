import torch
from genesis.configs.cfg import CFG as cfg
from genesis.core.dataset import DataModule

def test_pipeline():

    cfg["batch_size"] = 2
    cfg["num_workers"] = 0 
    cfg["prefetch_factor"] = None
    
    print("=== BƯỚC 1: KIỂM TRA ĐỌC DỮ LIỆU THẬT ===")
    try:
        data_manager = DataModule(cfg)
        loader = data_manager.build_loader()
        data_iter = iter(loader)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Dataset/Loader: {e}")
        return

    print("\n=== BƯỚC 2: TRÍCH XUẤT THỬ 1 BATCH ===")
    try:
        x, y = next(data_iter)
        print(f"👉 Shape của X (Đầu vào): {x.shape} (Batch_size, Block_size)")
        print(f"👉 Shape của Y (Nhãn):    {y.shape} (Batch_size, Block_size)")
        
        print("\n🔎 Chi tiết dữ liệu của mẫu đầu tiên trong batch:")
        print(f"X[0][:10] (10 tokens đầu): {x[0][:10].tolist()}")
        print(f"Y[0][:10] (10 tokens đầu): {y[0][:10].tolist()}")
        
        is_shifted = torch.equal(x[0][1:10], y[0][:9])
        print(f"👉 Kiểm tra logic dịch nhãn (X[1:10] == Y[:9]): {'HOÀN HẢO ✅' if is_shifted else 'SAI LOGIC ❌'}")
        
    except Exception as e:
        print(f"❌ Lỗi khi bốc dữ liệu từ vòng lặp: {e}")
        return

    print("\n=== BƯỚC 3: CHẠY THỬ MÔ HÌNH JARVIS (CPU) ===")
    try:
        from genesis.core.model import Genesis
        
        cfg["compile"] = False 
        
        print("Đang khởi tạo mô hình Genesis trên CPU...")
        model = Genesis(
            vocab_size=cfg["vocab_size"], dim=cfg["dim"],
            layers=cfg["layers"],         heads=cfg["heads"],
            block_size=cfg["block_size"],
        )
        model.train()
        
        print("Đang chạy thử 1 bước Forward Pass...")
        with torch.no_grad():
            logits = model(x)
            
        print(f"👉 Shape của đầu ra (Logits): {logits.shape} (Batch_size, Block_size, Vocab_size)")
        
        import torch.nn.functional as F
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(0, 1), ignore_index=-1)
        print(f"👉 Tính thử Loss thành công! Loss khởi tạo = {loss.item():.4f}")
        
        print("\n🎉 [KẾT LUẬN]: Dữ liệu thật hoàn toàn tương thích với kiến trúc mạng mạng!")
        
    except Exception as e:
        print(f"❌ Lỗi hệ thống hoặc mô hình không khớp với dữ liệu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()