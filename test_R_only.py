import pytest
from unittest.mock import patch

def test_window_alignment():
    """测试参考数据与训练窗口严格对齐"""
    # 模拟数据
    train_pos = np.array([100, 101, 102, 103, 104])
    ref_pos = np.array([100, 101, 102, 103, 104, 105])  # 包含额外位点
    
    with patch('allel.read_vcf') as mock_read:
        mock_read.return_value = {
            'variants/POS': ref_pos,
            'calldata/GT': np.zeros((6, 10, 2))  # 6个位点，10个样本
        }
        
        with pytest.raises(ValueError) as excinfo:
            dataset = RAGTrainDataset(...)
            dataset._build_faiss_indexes("dummy.vcf")
        
        assert "Position 104 not found" in str(excinfo.value)

def test_mask_consistency():
    """测试同窗口mask的一致性"""
    dataset = RAGTrainDataset(..., fixed_mask=True)
    
    # 获取同一窗口不同样本的mask
    mask1 = dataset[0]['mask'][:5].cpu().numpy()  # 假设窗口长度5
    mask2 = dataset[10]['mask'][:5].cpu().numpy()
    
    assert np.array_equal(mask1, mask2), "同一窗口的mask不一致"

def test_retrieval_accuracy():
    """端到端检索准确性测试"""
    # 构造测试数据
    ref_gt = np.array([
        [[[0,0], [1,1]], [[1,1], [0,0]]]  # 样本0: [[0,1], [1,0]]
    ], dtype=np.int8).transpose(2,3,1,0)  # 重塑为(位点, 样本, 单倍型)
    
    train_gt = np.array([[[0], [1]], [[1], [0]]], dtype=np.int8)  # 待填补样本
    
    with patch('h5py.File') as mock_h5:
        mock_h5.return_value = {
            'variants/POS': np.array([100, 101]),
            'calldata/GT': ref_gt
        }
        
        dataset = RAGTrainDataset(...)
        dataset.window_masks = [np.array([0,1])]  # 掩码第二个位点
        
        # 应检索到完全匹配的参考样本
        batch = [{'hap_1': torch.tensor([0,-1]), 'hap_2': torch.tensor([1,-1])}]
        collated = rag_collate_fn_with_dataset(batch, dataset)
        
        assert collated['rag_seg_h1'][0, 100] == 0  # 第一个位点
        assert collated['rag_seg_h2'][0, 101] == 0  # 第二个位点
