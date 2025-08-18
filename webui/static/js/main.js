/**
 * FactorMiner WebUI 主JavaScript文件
 */

// 全局变量
let currentAnalysis = null;
let charts = {};

// 工具函数
const utils = {
    showLoading: function(element, text = '加载中...') {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        if (element) {
            element.innerHTML = `<span class="loading"></span> ${text}`;
            element.disabled = true;
        }
    },

    hideLoading: function(element, originalText) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        if (element) {
            element.innerHTML = originalText;
            element.disabled = false;
        }
    },

    showNotification: function(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
        }
    },

    formatNumber: function(num, decimals = 4) {
        if (num === null || num === undefined) return '-';
        return Number(num).toFixed(decimals);
    },

    formatPercent: function(num, decimals = 2) {
        if (num === null || num === undefined) return '-';
        return (Number(num) * 100).toFixed(decimals) + '%';
    }
};

// API调用函数
const api = {
    call: async function(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };

            if (data && method !== 'GET') {
                options.body = JSON.stringify(data);
            }

            const response = await fetch(`/api${endpoint}`, options);
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || '请求失败');
            }

            return result;
        } catch (error) {
            console.error('API调用失败:', error);
            throw error;
        }
    },

    getFactorInfo: async function() {
        return await this.call('/factor-info');
    },

    loadData: async function(params) {
        return await this.call('/load-data', 'POST', params);
    },

    buildFactors: async function(params) {
        return await this.call('/build-factors', 'POST', params);
    },

    runAnalysis: async function(params) {
        return await this.call('/run-analysis', 'POST', params);
    }
};

// 页面初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('FactorMiner WebUI 初始化完成');
    
    // 初始化工具提示（简化版本）
    console.log('工具提示初始化完成');

    // 页面特定初始化
    const currentPage = window.location.pathname;
    
    if (currentPage === '/factors/builder') {
        initFactorBuilder();
    } else if (currentPage === '/factors/evaluation') {
        initFactorEvaluation();
    } else if (currentPage === '/factors/results') {
        initResults();
    }
});

// 因子构建页面初始化
function initFactorBuilder() {
    console.log('初始化因子构建页面');
    
    const form = document.getElementById('factorBuilderForm');
    if (form) {
        form.addEventListener('submit', handleFactorBuilderSubmit);
    }
}

// 因子评估页面初始化
function initFactorEvaluation() {
    console.log('初始化因子评估页面');
    
    const form = document.getElementById('factorEvaluationForm');
    if (form) {
        form.addEventListener('submit', handleFactorEvaluationSubmit);
    }
}

// 结果页面初始化
function initResults() {
    console.log('初始化结果页面');
    loadLatestResults();
}

// 处理因子构建表单提交
async function handleFactorBuilderSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    try {
        utils.showLoading(submitBtn, '构建因子中...');
        
        const formData = new FormData(form);
        const params = {
            symbol: formData.get('symbol'),
            timeframe: formData.get('timeframe'),
            factor_types: Array.from(form.querySelectorAll('input[name="factor_types"]:checked')).map(cb => cb.value),
            start_date: formData.get('start_date'),
            end_date: formData.get('end_date')
        };

        const result = await api.runAnalysis(params);
        
        if (result.success) {
            currentAnalysis = result;
            utils.showNotification('因子构建完成！', 'success');
            
            setTimeout(() => {
                window.location.href = '/factors/results';
            }, 1500);
        } else {
            utils.showNotification(result.error || '因子构建失败', 'danger');
        }
    } catch (error) {
        utils.showNotification('因子构建失败: ' + error.message, 'danger');
    } finally {
        utils.hideLoading(submitBtn, originalText);
    }
}

// 处理因子评估表单提交
async function handleFactorEvaluationSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    
    try {
        utils.showLoading(submitBtn, '评估因子中...');
        
        const formData = new FormData(form);
        const params = {
            symbol: formData.get('symbol'),
            timeframe: formData.get('timeframe'),
            evaluation_metrics: Array.from(form.querySelectorAll('input[name="evaluation_metrics"]:checked')).map(cb => cb.value)
        };

        const result = await api.runAnalysis(params);
        
        if (result.success) {
            currentAnalysis = result;
            utils.showNotification('因子评估完成！', 'success');
            
            setTimeout(() => {
                window.location.href = '/factors/results';
            }, 1500);
        } else {
            utils.showNotification(result.error || '因子评估失败', 'danger');
        }
    } catch (error) {
        utils.showNotification('因子评估失败: ' + error.message, 'danger');
    } finally {
        utils.hideLoading(submitBtn, originalText);
    }
}

// 加载最新结果
async function loadLatestResults() {
    try {
        const mockResults = {
            factors: [
                { name: '技术因子_1', ic: 0.15, ir: 0.8, win_rate: 0.65 },
                { name: '统计因子_1', ic: 0.12, ir: 0.7, win_rate: 0.62 },
                { name: 'ML因子_1', ic: 0.18, ir: 0.9, win_rate: 0.68 }
            ]
        };
        
        displayResults(mockResults);
    } catch (error) {
        utils.showNotification('加载结果失败: ' + error.message, 'danger');
    }
}

// 显示结果
function displayResults(results) {
    const resultsContainer = document.getElementById('resultsContainer');
    if (!resultsContainer) return;
    
    const factorsTable = document.getElementById('factorsTable');
    if (factorsTable && results.factors) {
        const tbody = factorsTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        results.factors.forEach(factor => {
            const row = tbody.insertRow();
            row.innerHTML = `
                <td>${factor.name}</td>
                <td>${utils.formatNumber(factor.ic)}</td>
                <td>${utils.formatNumber(factor.ir)}</td>
                <td>${utils.formatPercent(factor.win_rate)}</td>
                <td>
                    <span class="badge bg-${factor.ic > 0.1 ? 'success' : 'warning'}">
                        ${factor.ic > 0.1 ? '有效' : '一般'}
                    </span>
                </td>
            `;
        });
    }
}

// 全局函数导出
window.FactorMiner = {
    utils,
    api
};
