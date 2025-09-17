/**
 * 因子挖掘页面JavaScript
 */

// 全局变量
let localDataRows = [];
let rangeSlider = null;
let miningSession = null;
let progressInterval = null;
let progressEventSource = null;

// 步骤名称映射（后端步骤名称 -> 前端步骤ID）
const STEP_MAPPING = {
    'data_loading': 'step1',
    'factor_building': 'step2', 
    'factor_evaluation': 'step3',
    'factor_optimization': 'step4',
    'result_saving': 'step5'
};

// 步骤显示名称映射
const STEP_DISPLAY_NAMES = {
    'data_loading': '数据加载',
    'factor_building': '因子构建',
    'factor_evaluation': '因子评估',
    'factor_optimization': '因子优化',
    'result_saving': '结果保存'
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('因子挖掘页面初始化');
    initializePage();
    loadAlgorithms();
});

// 子进度统计（前端自估计）
let __mlSubStartTs = null;
let __mlLastPct = 0;
let __mlLastTs = 0;

/**
 * 初始化页面
 */
function initializePage() {
    console.log('开始初始化页面...');
    
    // 绑定事件监听器
    bindEventListeners();
    console.log('事件监听器绑定完成');
    
    // 初始化数据选择器
    initializeDataSelectors();
    console.log('数据选择器初始化完成');
    
    // 加载挖掘历史
    loadMiningHistory();
    console.log('挖掘历史加载完成');
    
    console.log('页面初始化完成');
}

/**
 * 绑定事件监听器
 */
function bindEventListeners() {
    // 表单提交
    const miningForm = document.getElementById('miningForm');
    if (miningForm) {
        miningForm.addEventListener('submit', handleMiningSubmit);
        console.log('挖掘表单事件监听器已绑定');
    } else {
        console.error('找不到挖掘表单元素');
    }
    
    // 数据选择器变化
    const exchangeSelect = document.getElementById('exchangeSelect');
    if (exchangeSelect) {
        exchangeSelect.addEventListener('change', refreshLocalMeta);
    }
    
    const tradeTypeSelect = document.getElementById('tradeTypeSelect');
    if (tradeTypeSelect) {
        tradeTypeSelect.addEventListener('change', refreshLocalMeta);
    }
    
    const symbolsSelect = document.getElementById('symbolsSelect');
    if (symbolsSelect) {
        symbolsSelect.addEventListener('change', updateTimeframesForSelection);
    }
    
    const timeframesSelect = document.getElementById('timeframesSelect');
    if (timeframesSelect) {
        timeframesSelect.addEventListener('change', updateRangeForSelection);
    }
}

/**
 * 初始化数据选择器
 */
function initializeDataSelectors() {
    console.log('初始化数据选择器...');
    
    // 初始化交易所选择器
    const exchangeSelect = document.getElementById('exchangeSelect');
    console.log('交易所选择器元素:', exchangeSelect);
    
    if (exchangeSelect) {
        exchangeSelect.innerHTML = '<option value="">请选择...</option>';
        exchangeSelect.innerHTML += '<option value="binance">Binance</option>';
        exchangeSelect.innerHTML += '<option value="okx">OKX</option>';
        exchangeSelect.innerHTML += '<option value="bybit">Bybit</option>';
        console.log('交易所选择器已初始化，选项数:', exchangeSelect.options.length);
    } else {
        console.error('找不到交易所选择器元素');
    }
    
    // 初始化交易类型选择器
    const tradeTypeSelect = document.getElementById('tradeTypeSelect');
    console.log('交易类型选择器元素:', tradeTypeSelect);
    
    if (tradeTypeSelect) {
        tradeTypeSelect.innerHTML = '<option value="">请选择...</option>';
        tradeTypeSelect.innerHTML += '<option value="futures">期货</option>';
        tradeTypeSelect.innerHTML += '<option value="spot">现货</option>';
        console.log('交易类型选择器已初始化，选项数:', tradeTypeSelect.options.length);
    } else {
        console.error('找不到交易类型选择器元素');
    }
    
    // 默认选择第一个选项
    if (exchangeSelect && exchangeSelect.options.length > 1) {
        exchangeSelect.selectedIndex = 1;
    }
    
    if (tradeTypeSelect && tradeTypeSelect.options.length > 1) {
        tradeTypeSelect.selectedIndex = 1;
    }
    
    // 延迟调用，确保DOM完全加载
    setTimeout(() => {
        if (exchangeSelect && exchangeSelect.value && tradeTypeSelect && tradeTypeSelect.value) {
            refreshLocalMeta();
        }
    }, 100);
}

/**
 * 刷新本地数据元信息
 */
async function refreshLocalMeta() {
    const exchangeSelect = document.getElementById('exchangeSelect');
    const tradeTypeSelect = document.getElementById('tradeTypeSelect');
    
    console.log('refreshLocalMeta 被调用');
    console.log('交易所选择:', exchangeSelect?.value);
    console.log('交易类型选择:', tradeTypeSelect?.value);
    
    if (!exchangeSelect || !tradeTypeSelect) {
        console.error('找不到交易所或交易类型选择器');
        return;
    }
    
    if (!exchangeSelect.value || !tradeTypeSelect.value) {
        console.log('交易所或交易类型未选择，跳过刷新');
        return;
    }
    
    try {
        console.log(`请求API: /api/data/local-data?exchange=${exchangeSelect.value}&trade_type=${tradeTypeSelect.value}`);
        const response = await fetch(`/api/data/local-data?exchange=${exchangeSelect.value}&trade_type=${tradeTypeSelect.value}`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('API返回数据:', data);
            localDataRows = data.data || [];
            console.log('解析后的数据行:', localDataRows);
            
            if (localDataRows.length > 0) {
                updateSymbolsSelect();
                updateTimeframesSelect();
                console.log('数据选择器更新完成');
            } else {
                console.log('没有找到数据');
                showAlert('warning', '没有找到可用的数据');
            }
        } else {
            console.error('API请求失败:', response.status, response.statusText);
            showAlert('error', `API请求失败: ${response.status}`);
        }
    } catch (error) {
        console.error('获取本地数据失败:', error);
        showAlert('error', '获取本地数据失败');
    }
}

/**
 * 更新交易对选择器
 */
function updateSymbolsSelect() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    if (!symbolsSelect) return;
    
    console.log('更新交易对选择器，数据行数:', localDataRows.length);
    
    // 获取唯一的交易对
    const symbols = [...new Set(localDataRows.map(row => row.symbol))];
    console.log('找到的交易对:', symbols);
    
    symbolsSelect.innerHTML = '<option value="">请选择...</option>';
    symbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol;
        option.textContent = symbol;
        symbolsSelect.appendChild(option);
    });
    
    console.log('交易对选择器已更新，选项数:', symbolsSelect.options.length);
}

/**
 * 更新时间框架选择器
 */
function updateTimeframesSelect() {
    const timeframesSelect = document.getElementById('timeframesSelect');
    if (!timeframesSelect) return;
    
    console.log('更新时间框架选择器，数据行数:', localDataRows.length);
    
    // 获取唯一的时间框架
    const timeframes = [...new Set(localDataRows.map(row => row.timeframe))];
    console.log('找到的时间框架:', timeframes);
    
    // 按时间长度排序
    const sortedTimeframes = timeframes.sort((a, b) => {
        const aMinutes = timeframeToMinutes(a);
        const bMinutes = timeframeToMinutes(b);
        return aMinutes - bMinutes;
    });
    
    console.log('排序后的时间框架:', sortedTimeframes);
    
    timeframesSelect.innerHTML = '<option value="">请选择...</option>';
    sortedTimeframes.forEach(timeframe => {
        const option = document.createElement('option');
        option.value = timeframe;
        option.textContent = timeframe;
        timeframesSelect.appendChild(option);
    });
    
    console.log('时间框架选择器已更新，选项数:', timeframesSelect.options.length);
}

/**
 * 时间框架转换为分钟数
 */
function timeframeToMinutes(timeframe) {
    const unit = timeframe.slice(-1);
    const value = parseInt(timeframe.slice(0, -1));
    
    switch (unit) {
        case 'm': return value;
        case 'h': return value * 60;
        case 'd': return value * 1440;
        default: return 0;
    }
}

/**
 * 更新交易对选择后的时间框架（使用包含式逻辑，参考因子评估页面）
 */
function updateTimeframesForSelection() {
    const timeframesSelect = document.getElementById('timeframesSelect');
    const selectedSymbols = Array.from(document.getElementById('symbolsSelect').selectedOptions).map(opt => opt.value);
    
    if (!localDataRows || localDataRows.length === 0) return;
    
    let timeframes = [];
    
    // 包含式（并集）展示：覆盖所选交易对的所有可用时间框架
    if (!selectedSymbols.length) {
        // 未选择交易对时，展示全部时间框架
        timeframes = [...new Set(localDataRows.map(r => r.timeframe))];
    } else {
        const union = new Set();
        selectedSymbols.forEach(sym => {
            localDataRows.filter(r => r.symbol === sym).forEach(r => union.add(r.timeframe));
        });
        timeframes = [...union];
    }
    
    // 按时间框架大小排序（从小到大）
    timeframes.sort((a, b) => timeframeToMinutes(a) - timeframeToMinutes(b));
    
    // 记录原选择，尽量保留
    const prev = new Set(Array.from(timeframesSelect.selectedOptions).map(opt => opt.value));
    
    timeframesSelect.innerHTML = '';
    timeframes.forEach(tf => {
        const option = document.createElement('option');
        option.value = tf;
        option.textContent = tf;
        if (prev.has(tf)) option.selected = true;
        timeframesSelect.appendChild(option);
    });
    
    // 更新时间范围（根据最新选择的交易对与时间框架）
    updateRangeForSelection();
}

/**
 * 根据当前选择（交易对/时间框架）动态更新可用时间范围滑条（参考因子评估页面）
 */
function updateRangeForSelection() {
    if (!localDataRows || localDataRows.length === 0) return;
    
    const selectedSymbols = Array.from(document.getElementById('symbolsSelect').selectedOptions).map(opt => opt.value);
    const selectedTimeframes = Array.from(document.getElementById('timeframesSelect').selectedOptions).map(opt => opt.value);
    
    let rows = localDataRows;
    
    if (selectedSymbols.length) {
        rows = rows.filter(r => selectedSymbols.includes(r.symbol));
    }
    
    if (selectedTimeframes.length) {
        rows = rows.filter(r => selectedTimeframes.includes(r.timeframe));
    }
    
    const starts = rows.map(r => r.date_range && r.date_range.start).filter(Boolean).map(s => new Date(s));
    const ends = rows.map(r => r.date_range && r.date_range.end).filter(Boolean).map(s => new Date(s));
    
    if (starts.length && ends.length) {
        const minDate = new Date(Math.min(...starts));
        const maxDate = new Date(Math.max(...ends));
        setupRangeSlider(minDate, maxDate);
    }
}

/**
 * 设置范围滑块（参考因子评估页面）
 */
function setupRangeSlider(startDate, endDate) {
    const container = document.getElementById('rangeSlider');
    const info = document.getElementById('rangeInfo');
    
    if (container.noUiSlider) {
        container.noUiSlider.destroy();
    }
    
    const totalMs = endDate - startDate;
    
    noUiSlider.create(container, {
        start: [0, 100],
        connect: true,
        step: 1,
        range: { min: 0, max: 100 }
    });
    
    const update = (values) => {
        const sv = parseInt(values[0], 10), ev = parseInt(values[1], 10);
        const s = new Date(startDate.getTime() + (sv/100)*totalMs);
        const e = new Date(startDate.getTime() + (ev/100)*totalMs);
        document.getElementById('startDate').value = s.toISOString().slice(0,10);
        document.getElementById('endDate').value = e.toISOString().slice(0,10);
        updateRangeInfo(`${s.toISOString().slice(0,10)} ~ ${e.toISOString().slice(0,10)}`);
    };
    
    container.noUiSlider.on('update', update);
    update([0, 100]);
}

/**
 * 更新范围信息显示
 */
function updateRangeInfo(message) {
    const rangeInfo = document.getElementById('rangeInfo');
    if (rangeInfo) {
        rangeInfo.textContent = message;
    }
}

/**
 * 处理挖掘表单提交
 */
async function handleMiningSubmit(event) {
    event.preventDefault();
    console.log('挖掘表单提交');
    
    try {
        // 获取表单数据
        const formData = new FormData(event.target);
        const miningData = {
            symbols: getSelectedValues('symbolsSelect'),
            timeframes: getSelectedValues('timeframesSelect'),
            selected_algorithms: getSelectedAlgorithms(),
            start_date: formData.get('startDate'),
            end_date: formData.get('endDate'),
            optimization_method: formData.get('optimizationMethod'),
            max_factors: parseInt(formData.get('maxFactors')) || 15,
            min_ic: parseFloat(formData.get('minIC')) || 0.02,
            min_ir: parseFloat(formData.get('minIR')) || 0.1,
            min_sample_size: parseInt(formData.get('minSampleSize')) || 30
        };
        
        console.log('挖掘配置:', miningData);
        
        // 验证数据
        if (miningData.symbols.length === 0) {
            showAlert('error', '请选择至少一个交易对');
            return;
        }
        
        if (miningData.timeframes.length === 0) {
            showAlert('error', '请选择至少一个时间框架');
            return;
        }
        
        if (miningData.selected_algorithms.length === 0) {
            showAlert('error', '请选择至少一个算法');
            return;
        }
        
        // 更新按钮状态
        updateStartButton(true);
        
        // 显示等待状态
        showWaitingState();
        
        // 启动挖掘
        const response = await fetch('/api/mining/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(miningData)
        });
        
        const result = await response.json();
        console.log('挖掘启动结果:', result);
        
        if (result.success) {
            // 保存会话ID
            miningSession = result.session_id;
            
            // 显示预估时间信息
            if (result.estimated_time) {
                const estimatedTime = formatTime(result.estimated_time);
                showAlert('info', `挖掘已启动！预估总时间: ${estimatedTime}`);
                
                // 更新预估时间显示
                const timeDisplay = document.getElementById('timeDisplay');
                if (timeDisplay) {
                    timeDisplay.textContent = `预估总时间: ${estimatedTime}`;
                }
                
                // 更新系统信息显示
                if (result.system_info) {
                    const systemDisplay = document.getElementById('systemDisplay');
                    if (systemDisplay) {
                        const { cpu_count, memory_gb, memory_percent } = result.system_info;
                        systemDisplay.textContent = `CPU: ${cpu_count}核 | 内存: ${memory_gb}GB (${memory_percent}%)`;
                    }
                }
            }
            
            // 显示进度界面
            showMiningProgress();
            
            // 启动进度监控
            startProgressMonitoring(result.session_id);
            
        } else {
            // 挖掘失败
            showAlert('error', result.error || '挖掘启动失败');
        }
        
    } catch (error) {
        console.error('挖掘失败:', error);
        showAlert('error', `挖掘过程中发生错误: ${error.message}`);
    } finally {
        // 恢复按钮状态
        updateStartButton(false);
    }
}

/**
 * 获取选择框的选中值
 */
function getSelectedValues(selectId) {
    if (selectId === 'factorTypes') {
        // 特殊处理因子类型复选框（已废弃，保留兼容性）
        const checkboxes = document.querySelectorAll('input[name="factorType"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }
    
    const select = document.getElementById(selectId);
    if (!select) return [];
    
    const selectedOptions = Array.from(select.selectedOptions);
    return selectedOptions.map(option => option.value).filter(value => value);
}

/**
 * 验证挖掘表单
 */
function validateMiningForm() {
    const symbols = getSelectedValues('symbolsSelect');
    const timeframes = getSelectedValues('timeframesSelect');
    const selectedAlgorithms = getSelectedAlgorithms();
    
    if (symbols.length === 0) {
        showAlert('error', '请选择至少一个交易对');
        return false;
    }
    
    if (timeframes.length === 0) {
        showAlert('error', '请选择至少一个时间框架');
        return false;
    }
    
    if (selectedAlgorithms.length === 0) {
        showAlert('error', '请选择至少一个算法');
        return false;
    }
    
    return true;
}

/**
 * 显示等待状态
 */
function showWaitingState() {
    const waitingState = document.getElementById('waitingState');
    if (waitingState) {
        waitingState.style.display = 'block';
    }
    
    // 隐藏其他状态
    const miningProgress = document.getElementById('miningProgress');
    if (miningProgress) {
        miningProgress.style.display = 'none';
    }
    
    const miningResults = document.getElementById('miningResults');
    if (miningResults) {
        miningResults.style.display = 'none';
    }
}

/**
 * 获取挖掘表单数据
 */
function getMiningFormData() {
    const exchange = document.getElementById('exchangeSelect').value;
    const tradeType = document.getElementById('tradeTypeSelect').value;
    const symbols = Array.from(document.getElementById('symbolsSelect').selectedOptions).map(opt => opt.value);
    const timeframes = Array.from(document.getElementById('timeframesSelect').selectedOptions).map(opt => opt.value);
    const maxFactors = parseInt(document.getElementById('maxFactors').value) || 15;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    return {
        exchange: exchange,
        trade_type: tradeType,
        symbols: symbols,
        timeframes: timeframes,
        max_factors: maxFactors,
        start_date: startDate,
        end_date: endDate,
        optimization_method: 'greedy'
    };
}


/**
 * 开始因子挖掘
 */
async function startMining(formData) {
    try {
        console.log('开始因子挖掘...');
        
        // 显示进度界面
        showMiningProgress();
        
        // 更新按钮状态
        updateStartButton(true);
        
        // 调用挖掘API
        const response = await fetch('/api/mining/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('挖掘API响应:', result);
        
        if (result.success) {
            // 挖掘成功，保存会话ID
            miningSession = result;
            showAlert('success', '因子挖掘已启动，正在监控进度...');
            console.log('挖掘会话ID:', result.session_id);
            
            // 添加调试信息
            addDebugInfo('挖掘任务已启动', 'success');
            addDebugInfo(`会话ID: ${result.session_id}`, 'info');
            
            // 重置进度步骤
            resetProgressSteps();
            
            // 启动进度监控
            startProgressMonitoring(result.session_id);
        } else {
            // 挖掘失败
            addDebugInfo(`挖掘启动失败: ${result.error || '未知错误'}`, 'error');
            showAlert('error', result.error || '挖掘启动失败');
        }
        
    } catch (error) {
        console.error('挖掘失败:', error);
        showAlert('error', `挖掘过程中发生错误: ${error.message}`);
    } finally {
        // 恢复按钮状态
        updateStartButton(false);
    }
}

/**
 * 显示挖掘进度界面
 */
function showMiningProgress() {
    // 隐藏等待状态
    const waitingState = document.getElementById('waitingState');
    if (waitingState) {
        waitingState.style.display = 'none';
    }
    
    // 显示进度界面
    const miningProgress = document.getElementById('miningProgress');
    if (miningProgress) {
        miningProgress.style.display = 'block';
    }
    
    // 重置子进度条
    resetSubProgress();
}

/**
 * 重置进度步骤
 */
function resetProgressSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
    console.log('重置进度步骤:', steps);
    
    steps.forEach(stepId => {
        const stepElement = document.getElementById(stepId);
        if (stepElement) {
            const progressBar = stepElement.querySelector('.progress-fill');
            const stepIcon = stepElement.querySelector('.step-icon');
            const stepDetails = stepElement.querySelector('.step-details');
            
            if (progressBar) {
                progressBar.style.width = '0%';
                console.log(`重置进度条 ${stepId}: 0%`);
            }
            if (stepIcon) {
                stepIcon.className = 'step-icon pending';
                stepIcon.innerHTML = '<i class="fas fa-clock"></i>';
                console.log(`重置图标 ${stepId}: 等待状态`);
            }
            if (stepDetails) {
                stepDetails.textContent = '等待开始...';
                console.log(`重置详情 ${stepId}: 等待开始...`);
            }
        } else {
            console.error(`找不到步骤元素: ${stepId}`);
        }
    });
    
    // 重置总体进度
    updateOverallProgress(0, null, [], null, null);
}

/**
 * 启动进度监控
 */
function startProgressMonitoring(sessionId) {
    console.log('启动进度监控，会话ID:', sessionId);
    
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    if (progressEventSource) {
        progressEventSource.close();
    }
    
    // 使用Server-Sent Events进行实时进度更新
    progressEventSource = new EventSource(`/api/mining/progress/${sessionId}`);
    
    progressEventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('收到进度更新:', data);
            
            if (data.success !== false) {
                updateProgressDisplay(data);
                
                if (data.status === 'completed') {
                    // 挖掘完成，获取完整结果
                    handleMiningCompleted(sessionId);
                } else if (data.status === 'error') {
                    handleMiningError(data.error);
                }
            } else {
                addDebugInfo(`进度更新失败: ${data.error || '未知错误'}`, 'error');
            }
        } catch (error) {
            console.error('解析进度数据失败:', error);
            addDebugInfo(`解析进度数据失败: ${error.message}`, 'error');
        }
    };
    
    progressEventSource.onerror = function(error) {
        console.error('进度流错误:', error);
        addDebugInfo('SSE连接失败，切换到轮询模式', 'warning');
        // 如果SSE失败，回退到轮询
        fallbackToPolling(sessionId);
    };
}

/**
 * 回退到轮询方式
 */
function fallbackToPolling(sessionId) {
    console.log('回退到轮询方式');
    addDebugInfo('开始轮询模式监控进度', 'info');
    
    if (progressEventSource) {
        progressEventSource.close();
        progressEventSource = null;
    }
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/mining/status/${sessionId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    updateProgressDisplay(data);
                    
                    if (data.status === 'completed') {
                        // 挖掘完成，获取完整结果
                        handleMiningCompleted(sessionId);
                    } else if (data.status === 'error') {
                        handleMiningError(data.error);
                    }
                } else {
                    addDebugInfo(`轮询获取状态失败: ${data.error || '未知错误'}`, 'error');
                }
            } else {
                addDebugInfo(`轮询HTTP错误: ${response.status}`, 'error');
            }
        } catch (error) {
            console.error('获取进度失败:', error);
            addDebugInfo(`轮询获取进度失败: ${error.message}`, 'error');
        }
    }, 1000); // 每秒更新一次
}

/**
 * 更新进度显示（简化版）
 */
function updateProgressDisplay(data) {
    const { progress, current_step, messages } = data;
    console.log('更新进度显示:', { progress, current_step, messages });
    
    // 更新总体进度
    const overallProgress = progress || 0;
    updateOverallProgress(overallProgress, current_step, messages);
    
    // 添加调试信息
    addDebugInfo(`[${new Date().toLocaleTimeString()}] 进度更新: ${overallProgress}% - ${current_step || '未知步骤'}`);
    
    // 如果有消息，添加到调试信息
    if (messages && messages.length > 0) {
        messages.forEach(msg => {
            addDebugInfo(`[${new Date().toLocaleTimeString()}] ${msg.message || msg}`);
        });
    }
}

/**
 * 更新总体进度（简化版）
 */
function updateOverallProgress(progress, currentStep, messages) {
    const overallProgressBar = document.getElementById('overallProgress');
    const progressPercent = document.getElementById('progressPercent');
    const progressBadge = document.getElementById('progressBadge');
    
    if (overallProgressBar) {
        overallProgressBar.style.width = `${progress}%`;
    }
    
    if (progressPercent) {
        progressPercent.textContent = `${progress}%`;
    }
    
    if (progressBadge) {
        if (progress === 0) {
            progressBadge.textContent = '准备中';
            progressBadge.className = 'badge bg-secondary ms-2';
        } else if (progress === 100) {
            progressBadge.textContent = '完成';
            progressBadge.className = 'badge bg-success ms-2';
        } else {
            progressBadge.textContent = '运行中';
            progressBadge.className = 'badge bg-primary ms-2';
        }
    }
}

/**
 * 添加调试信息
 */
function addDebugInfo(message, type = 'info') {
    const debugInfo = document.getElementById('debugInfo');
    if (!debugInfo) return;
    
    const debugItem = document.createElement('div');
    debugItem.className = `debug-item ${type}`;
    
    const time = new Date().toLocaleTimeString();
    debugItem.innerHTML = `
        <span class="debug-time">[${time}]</span>
        <span class="debug-message">${message}</span>
    `;
    
    debugInfo.appendChild(debugItem);
    
    // 限制调试信息数量，避免过多
    const items = debugInfo.querySelectorAll('.debug-item');
    if (items.length > 50) {
        items[0].remove();
    }
    
    // 自动滚动到底部
    debugInfo.scrollTop = debugInfo.scrollHeight;
}

/**
 * 清空调试信息
 */
function clearDebugInfo() {
    const debugInfo = document.getElementById('debugInfo');
    if (debugInfo) {
        debugInfo.innerHTML = `
            <div class="debug-item">
                <span class="debug-time">[${new Date().toLocaleTimeString()}]</span>
                <span class="debug-message">调试信息已清空</span>
            </div>
        `;
    }
}

/**
 * 格式化时间显示
 */
function formatTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)}秒`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        return `${minutes}分${remainingSeconds}秒`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const remainingMinutes = Math.floor((seconds % 3600) / 60);
        return `${hours}小时${remainingMinutes}分`;
    }
}

/**
 * 处理挖掘完成
 */
async function handleMiningCompleted(sessionId) {
    console.log('挖掘完成，会话ID:', sessionId);
    
    // 添加调试信息
    addDebugInfo('挖掘任务完成！', 'success');
    
    // 停止进度监控
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    if (progressEventSource) {
        progressEventSource.close();
        progressEventSource = null;
    }
    
    // 更新按钮状态
    updateStartButton(false);
    
    // 显示完成提示
    showAlert('success', '因子挖掘完成！');
    
    try {
        // 获取完整的挖掘结果
        addDebugInfo('正在获取挖掘结果...', 'info');
        console.log('获取完整挖掘结果...');
        const response = await fetch(`/api/mining/result/${sessionId}`);
        if (response.ok) {
            // 获取响应文本
            const responseText = await response.text();
            console.log('挖掘结果响应文本:', responseText);
            
            // 清理NaN值，替换为null
            const cleanedText = responseText.replace(/: NaN/g, ': null');
            console.log('清理后的响应文本:', cleanedText);
            
            let resultData;
            try {
                // 尝试解析清理后的JSON
                resultData = JSON.parse(cleanedText);
            } catch (parseError) {
                console.error('JSON解析失败:', parseError);
                console.error('清理后的响应文本:', cleanedText);
                addDebugInfo(`JSON解析失败: ${parseError.message}`, 'error');
                throw new Error(`JSON解析失败: ${parseError.message}`);
            }
            
            console.log('挖掘结果数据:', resultData);
            
            if (resultData.success !== false) {
                // 显示结果
                addDebugInfo('挖掘结果获取成功，正在显示...', 'success');
                showMiningResults(resultData);
                // 追加加载对比报告
                loadDiffReport(sessionId);
            } else {
                console.error('获取挖掘结果失败:', resultData.error);
                addDebugInfo(`获取挖掘结果失败: ${resultData.error}`, 'error');
                showAlert('error', `获取挖掘结果失败: ${resultData.error}`);
            }
        } else {
            addDebugInfo(`HTTP错误: ${response.status}`, 'error');
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    } catch (error) {
        console.error('获取挖掘结果失败:', error);
        addDebugInfo(`获取挖掘结果失败: ${error.message}`, 'error');
        showAlert('error', `获取挖掘结果失败: ${error.message}`);
    }
    
    // 刷新挖掘历史
    loadMiningHistory();
}

/**
 * 处理挖掘错误
 */
function handleMiningError(error) {
    console.error('挖掘错误:', error);
    
    // 添加调试信息
    addDebugInfo(`挖掘任务失败: ${error}`, 'error');
    
    // 停止进度监控
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    if (progressEventSource) {
        progressEventSource.close();
        progressEventSource = null;
    }
    
    // 更新按钮状态
    updateStartButton(false);
    
    // 显示错误提示
    showAlert('error', `挖掘失败: ${error}`);
}

/**
 * 显示挖掘结果
 */
function showMiningResults(data) {
    // 隐藏进度界面
    const miningProgress = document.getElementById('miningProgress');
    if (miningProgress) {
        miningProgress.style.display = 'none';
    }
    
    // 显示结果界面
    const miningResults = document.getElementById('miningResults');
    if (miningResults) {
        miningResults.style.display = 'block';
        updateResultsOverview(data);
        updateResultsTable(data);
        // 准备对比报告容器
        ensureDiffContainer();
    }
}

/**
 * 确保对比报告容器存在
 */
function ensureDiffContainer() {
    const miningResults = document.getElementById('miningResults');
    if (!miningResults) return;
    let diffSection = document.getElementById('diffSection');
    if (!diffSection) {
        diffSection = document.createElement('div');
        diffSection.id = 'diffSection';
        diffSection.className = 'mt-4';
        diffSection.innerHTML = `
            <h5>因子对比报告</h5>
            <div id="diffSummary" class="mb-2 small text-muted"></div>
            <div id="diffTableWrap" class="table-responsive"></div>
            <div class="mt-2">
                <button id="saveSelectedBtn" class="btn btn-primary btn-sm" disabled>保存选中因子到因子库</button>
                <span id="saveSelectedHint" class="ms-2 text-muted small"></span>
            </div>
        `;
        miningResults.appendChild(diffSection);
        const btn = diffSection.querySelector('#saveSelectedBtn');
        if (btn) {
            btn.addEventListener('click', onSaveSelectedFactors);
        }
    }
}

/**
 * 加载对比报告
 */
async function loadDiffReport(sessionId) {
    try {
        const res = await fetch(`/api/mining/diff/${sessionId}`);
        if (!res.ok) {
            console.warn('diff接口响应非200', res.status);
            return;
        }
        const payload = await res.json();
        if (payload && payload.success) {
            console.log('diff payload:', payload);
            renderDiffReport(sessionId, payload.diff_report || {});
        }
    } catch (e) {
        console.warn('加载对比报告失败', e);
    }
}

/**
 * 渲染对比报告
 */
function renderDiffReport(sessionId, diff) {
    ensureDiffContainer();
    const summaryEl = document.getElementById('diffSummary');
    const tableWrap = document.getElementById('diffTableWrap');
    const saveBtn = document.getElementById('saveSelectedBtn');
    if (!summaryEl || !tableWrap) return;

    const summary = diff.summary || { total_mined: 0, new: 0, identical: 0, different: 0, missing_artifact: 0 };
    summaryEl.textContent = `总计: ${summary.total_mined}，新增: ${summary.new}，差异: ${summary.different}，相同: ${summary.identical}，缺少模型: ${summary.missing_artifact}`;

    const items = diff.items || [];
    if (!items.length) {
        tableWrap.innerHTML = '<div class="alert alert-secondary">暂无对比数据</div>';
        if (saveBtn) saveBtn.disabled = true;
        return;
    }

    const table = document.createElement('table');
    table.className = 'table table-sm table-striped align-middle';
    table.innerHTML = `
        <thead>
            <tr>
                <th style="width: 32px;"><input type="checkbox" id="diffSelectAll"></th>
                <th>因子ID</th>
                <th>状态</th>
                <th>现有模型签名/函数签名</th>
                <th>新模型签名</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;
    const tbody = table.querySelector('tbody');

    items.forEach(it => {
        const tr = document.createElement('tr');
        const existingSig = (it.existing && (it.existing.model_meta && it.existing.model_meta.signature)) || it.existing.function_signature || '-';
        const newSig = (it.new && it.new.model_meta && it.new.model_meta.signature) || '-';
        const selectable = (it.status === 'new' || it.status === 'different');
        tr.innerHTML = `
            <td><input type="checkbox" class="diffRowChk" data-factor-id="${it.factor_id}" ${selectable ? '' : 'disabled'}></td>
            <td>${it.factor_id}</td>
            <td><span class="badge ${badgeClassForStatus(it.status)}">${labelForStatus(it.status)}</span></td>
            <td><code>${escapeHtml(String(existingSig))}</code></td>
            <td><code>${escapeHtml(String(newSig))}</code></td>
        `;
        tbody.appendChild(tr);
    });

    tableWrap.innerHTML = '';
    tableWrap.appendChild(table);

    // 绑定全选与勾选变化
    const selectAll = table.querySelector('#diffSelectAll');
    const rowChks = table.querySelectorAll('.diffRowChk');
    const updateBtn = () => {
        const anySelected = Array.from(rowChks).some(chk => chk.checked && !chk.disabled);
        if (saveBtn) {
            saveBtn.disabled = !anySelected;
            saveBtn.dataset.sessionId = sessionId;
        }
        const hint = document.getElementById('saveSelectedHint');
        if (hint) {
            const selCount = Array.from(rowChks).filter(chk => chk.checked && !chk.disabled).length;
            hint.textContent = selCount ? `已选择 ${selCount} 个因子` : '';
        }
    };
    if (selectAll) {
        selectAll.addEventListener('change', () => {
            rowChks.forEach(chk => { if (!chk.disabled) chk.checked = selectAll.checked; });
            updateBtn();
        });
    }
    rowChks.forEach(chk => chk.addEventListener('change', updateBtn));
    updateBtn();
}

function badgeClassForStatus(status) {
    switch (status) {
        case 'new': return 'bg-success';
        case 'different': return 'bg-warning text-dark';
        case 'identical': return 'bg-secondary';
        case 'missing_artifact': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

function labelForStatus(status) {
    switch (status) {
        case 'new': return '新增';
        case 'different': return '差异';
        case 'identical': return '相同';
        case 'missing_artifact': return '缺少模型';
        default: return status;
    }
}

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

async function onSaveSelectedFactors() {
    const btn = document.getElementById('saveSelectedBtn');
    if (!btn || btn.disabled) return;
    const sessionId = btn.dataset.sessionId;
    const wrap = document.getElementById('diffTableWrap');
    if (!wrap) return;
    const chks = wrap.querySelectorAll('.diffRowChk');
    const selected = Array.from(chks).filter(chk => chk.checked && !chk.disabled).map(chk => chk.dataset.factorId);
    if (!selected.length) return;
    try {
        btn.disabled = true;
        const resp = await fetch('/api/mining/save_selected_factors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, factor_ids: selected })
        });
        const json = await resp.json();
        if (json && json.success) {
            showAlert('success', `已保存 ${json.saved_count} 个因子定义`);
        } else {
            showAlert('error', `保存失败: ${json && json.message ? json.message : '未知错误'}`);
        }
    } catch (e) {
        showAlert('error', `保存失败: ${e.message}`);
    } finally {
        btn.disabled = false;
    }
}

/**
 * 更新结果概览
 */
function updateResultsOverview(data) {
    console.log('更新结果概览:', data);
    
    try {
        // 更新总因子数
        const totalFactorsElement = document.getElementById('totalFactors');
        if (totalFactorsElement) {
            totalFactorsElement.textContent = data.factor_count || data.factors_info?.total_factors || 0;
        }
        
        // 更新选中因子数
        const selectedFactorsElement = document.getElementById('selectedFactors');
        if (selectedFactorsElement && data.optimization && data.optimization.selected_factors) {
            selectedFactorsElement.textContent = data.optimization.selected_factors.length || 0;
        }
        
        // 更新平均IC
        const avgICElement = document.getElementById('avgIC');
        if (avgICElement) {
            const evaluationData = data.evaluation?.evaluation || data.evaluation;
            if (evaluationData) {
                const factors = Object.values(evaluationData);
                if (factors.length > 0) {
                    const avgIC = factors.reduce((sum, factor) => {
                        const ic = factor.ic_pearson || factor.ic_spearman || 0;
                        return sum + ic;
                    }, 0) / factors.length;
                    avgICElement.textContent = avgIC.toFixed(4);
                }
            }
        }
        
        // 更新平均多空收益
        const avgReturnElement = document.getElementById('avgReturn');
        if (avgReturnElement) {
            const evaluationData = data.evaluation?.evaluation || data.evaluation;
            if (evaluationData) {
                const factors = Object.values(evaluationData);
                if (factors.length > 0) {
                    const avgReturn = factors.reduce((sum, factor) => {
                        return sum + (factor.long_short_return || 0);
                    }, 0) / factors.length;
                    avgReturnElement.textContent = `${(avgReturn * 100).toFixed(2)}%`;
                }
            }
        }
        
        console.log('结果概览更新完成');
    } catch (error) {
        console.error('更新结果概览失败:', error);
    }
}

/**
 * 更新结果表格
 */
function updateResultsTable(data) {
    console.log('更新结果表格:', data);
    
    try {
        // 查找结果表格容器
        const resultsTableContainer = document.querySelector('#miningResults .table-responsive');
        if (!resultsTableContainer) {
            console.error('找不到结果表格容器');
            return;
        }
        
        // 清空现有内容
        resultsTableContainer.innerHTML = '';
        
        // 获取实际的评估数据（支持嵌套结构）
        const evaluationData = data.evaluation?.evaluation || data.evaluation;
        if (!evaluationData || Object.keys(evaluationData).length === 0) {
            resultsTableContainer.innerHTML = '<div class="alert alert-warning">暂无挖掘结果</div>';
            return;
        }
        
        // 创建表格
        const table = document.createElement('table');
        table.className = 'table table-striped table-hover';
        
        // 创建表头
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>因子名称</th>
                <th>多空收益</th>
                <th>IC (Pearson)</th>
                <th>IC (Spearman)</th>
                <th>胜率</th>
                <th>夏普比率</th>
                <th>类型</th>
            </tr>
        `;
        table.appendChild(thead);
        
        // 创建表体
        const tbody = document.createElement('tbody');
        
        // 获取算法信息（兼容旧格式）
        let algorithmType = '未知';
        if (data.config?.selected_algorithms && data.config.selected_algorithms.length > 0) {
            algorithmType = data.config.selected_algorithms[0];
        } else if (data.factors_info?.factor_types && data.factors_info.factor_types.length > 0) {
            // 兼容旧格式，从factor_types获取
            algorithmType = data.factors_info.factor_types[0];
        }
        
        // 按多空收益排序（降序）
        const sortedFactors = Object.entries(evaluationData).sort(([,a], [,b]) => {
            const returnA = a.long_short_return || 0;
            const returnB = b.long_short_return || 0;
            return returnB - returnA;
        });
        
        sortedFactors.forEach(([factorName, factorData]) => {
            const row = document.createElement('tr');
            // 计算多空收益率（百分比）
            const longShortReturn = (factorData.long_short_return || 0) * 100;
            // 根据多空收益设置行样式
            const returnClass = longShortReturn > 0 ? 'text-success' : longShortReturn < 0 ? 'text-danger' : '';
            row.innerHTML = `
                <td>${factorName}</td>
                <td class="${returnClass}"><strong>${longShortReturn.toFixed(2)}%</strong></td>
                <td>${(factorData.ic_pearson || 0).toFixed(4)}</td>
                <td>${(factorData.ic_spearman || 0).toFixed(4)}</td>
                <td>${((factorData.win_rate || 0) * 100).toFixed(1)}%</td>
                <td>${(factorData.sharpe_ratio || 0).toFixed(2)}</td>
                <td><span class="badge bg-secondary">${algorithmType}</span></td>
            `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        
        // 添加到容器
        resultsTableContainer.appendChild(table);
        
        // 更新图表
        updateCharts(data);
        
        console.log('结果表格更新完成');
    } catch (error) {
        console.error('更新结果表格失败:', error);
    }
}

/**
 * 更新图表
 */
function updateCharts(data) {
    try {
        console.log('开始更新图表:', data);
        
        // 更新类型分布图
        updateTypeChart(data);
        
        // 更新性能分布图
        updatePerformanceChart(data);
        
        console.log('图表更新完成');
    } catch (error) {
        console.error('更新图表失败:', error);
    }
}

/**
 * 更新类型分布图
 */
function updateTypeChart(data) {
    try {
        const canvas = document.getElementById('typeChart');
        if (!canvas) {
            console.error('找不到类型分布图容器');
            return;
        }
        
        // 获取算法信息（兼容旧格式）
        let algorithmType = '未知';
        if (data.config?.selected_algorithms && data.config.selected_algorithms.length > 0) {
            algorithmType = data.config.selected_algorithms[0];
        } else if (data.factors_info?.factor_types && data.factors_info.factor_types.length > 0) {
            // 兼容旧格式，从factor_types获取
            algorithmType = data.factors_info.factor_types[0];
        }
        
        // 统计各算法的因子数量
        const typeCounts = {};
        typeCounts[algorithmType] = Object.keys(data.evaluation || {}).length;
        
        // 销毁旧图表
        if (canvas._chartInstance) {
            canvas._chartInstance.destroy();
        }
        
        // 创建新图表
        const ctx = canvas.getContext('2d');
        canvas._chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(typeCounts),
                datasets: [{
                    data: Object.values(typeCounts),
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: '因子类型分布',
                        font: { size: 14 }
                    }
                }
            }
        });
        
        console.log('类型分布图更新完成');
    } catch (error) {
        console.error('更新类型分布图失败:', error);
    }
}

/**
 * 更新性能分布图
 */
function updatePerformanceChart(data) {
    try {
        const canvas = document.getElementById('performanceChart');
        if (!canvas) {
            console.error('找不到性能分布图容器');
            return;
        }
        
        if (!data.evaluation || Object.keys(data.evaluation).length === 0) {
            return;
        }
        
        // 提取IC值用于性能分布
        const icValues = [];
        Object.values(data.evaluation).forEach(factorData => {
            if (factorData.ic_pearson !== undefined) {
                icValues.push(Math.abs(factorData.ic_pearson));
            }
        });
        
        if (icValues.length === 0) {
            return;
        }
        
        // 计算性能分布区间
        const maxIC = Math.max(...icValues);
        const minIC = Math.min(...icValues);
        const range = maxIC - minIC;
        const binCount = 5;
        const binSize = range / binCount;
        
        const bins = new Array(binCount).fill(0);
        const binLabels = [];
        
        for (let i = 0; i < binCount; i++) {
            const start = minIC + i * binSize;
            const end = minIC + (i + 1) * binSize;
            binLabels.push(`${start.toFixed(4)}-${end.toFixed(4)}`);
            
            icValues.forEach(ic => {
                if (ic >= start && ic < end) {
                    bins[i]++;
                }
            });
        }
        
        // 处理最后一个区间（包含最大值）
        bins[binCount - 1] += icValues.filter(ic => ic === maxIC).length;
        
        // 销毁旧图表
        if (canvas._chartInstance) {
            canvas._chartInstance.destroy();
        }
        
        // 创建新图表
        const ctx = canvas.getContext('2d');
        canvas._chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: '因子数量',
                    data: bins,
                    backgroundColor: 'rgba(75, 192, 192, 0.8)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'IC值分布',
                        font: { size: 14 }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '因子数量'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'IC值范围'
                        }
                    }
                }
            }
        });
        
        console.log('性能分布图更新完成');
    } catch (error) {
        console.error('更新性能分布图失败:', error);
    }
}

/**
 * 更新开始按钮状态
 */
function updateStartButton(disabled) {
    const startBtn = document.getElementById('startMiningBtn');
    if (startBtn) {
        startBtn.disabled = disabled;
        startBtn.innerHTML = disabled ? 
            '<i class="fas fa-spinner fa-spin me-2"></i>挖掘中...' : 
            '<i class="fas fa-rocket me-2"></i>开始因子挖掘';
    }
}

/**
 * 加载挖掘历史
 */
async function loadMiningHistory() {
    try {
        console.log('开始加载挖掘历史...');
        
        // 检查网络连接
        if (!navigator.onLine) {
            console.error('网络连接不可用');
            return;
        }
        
        const response = await fetch('/api/mining/history');
        console.log('API响应:', response);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // 获取响应文本
        const responseText = await response.text();
        console.log('API响应文本:', responseText);
        
        // 清理NaN值，替换为null
        const cleanedText = responseText.replace(/: NaN/g, ': null');
        console.log('清理后的响应文本:', cleanedText);
        
        let data;
        try {
            // 尝试解析清理后的JSON
            data = JSON.parse(cleanedText);
        } catch (parseError) {
            console.error('JSON解析失败:', parseError);
            console.error('清理后的响应文本:', cleanedText);
            throw new Error(`JSON解析失败: ${parseError.message}`);
        }
        
        console.log('挖掘历史数据:', data);
        
        if (data.success) {
            console.log(`成功获取历史数据，共 ${data.sessions?.length || 0} 个会话`);
            updateHistoryTable(data.sessions);
        } else {
            console.error('加载挖掘历史失败:', data.error);
            // 显示错误信息
            showAlert('error', `加载挖掘历史失败: ${data.error}`);
        }
    } catch (error) {
        console.error('加载挖掘历史失败:', error);
        // 显示错误信息
        showAlert('error', `加载挖掘历史失败: ${error.message}`);
        
        // 尝试显示空状态
        const historyTableBody = document.getElementById('historyTableBody');
        if (historyTableBody) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-muted">
                        <i class="fas fa-exclamation-triangle me-2"></i>加载失败: ${error.message}
                    </td>
                </tr>
            `;
        }
    }
}

/**
 * 更新历史表格
 */
function updateHistoryTable(sessions) {
    console.log('开始更新历史表格，数据:', sessions);
    
    const historyTableBody = document.getElementById('historyTableBody');
    if (!historyTableBody) {
        console.error('找不到历史表格元素');
        showAlert('error', '找不到历史表格元素');
        return;
    }
    
    console.log('找到历史表格元素，开始更新...');
    
    // 清空现有内容
    historyTableBody.innerHTML = '';
    
    if (!sessions || sessions.length === 0) {
        console.log('没有历史数据，显示空状态');
        historyTableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-muted">
                    <i class="fas fa-info-circle me-2"></i>暂无挖掘历史
                </td>
            </tr>
        `;
        return;
    }
    
    console.log(`开始处理 ${sessions.length} 个会话...`);
    
    // 添加历史记录
    sessions.forEach((session, index) => {
        try {
            console.log(`处理第 ${index + 1} 个会话:`, session);
            
            const row = document.createElement('tr');
            
            // 安全地获取会话ID
            const sessionId = session.session_id || `unknown-${index}`;
            
            // 格式化时间
            console.log('原始时间戳:', session.timestamp);
            let timeStr = '时间无效';
            
            if (session.timestamp) {
                try {
                    const timestamp = new Date(session.timestamp);
                    console.log('解析后的时间对象:', timestamp);
                    console.log('时间是否有效:', !isNaN(timestamp.getTime()));
                    
                    if (!isNaN(timestamp.getTime())) {
                        timeStr = timestamp.toLocaleString('zh-CN');
                    }
                } catch (timeError) {
                    console.error('时间格式化失败:', timeError);
                }
            }
            
            // 安全地获取配置信息
            const config = session.config || {};
            const symbols = Array.isArray(config.symbols) ? config.symbols : [];
            const timeframes = Array.isArray(config.timeframes) ? config.timeframes : [];
            const selectedAlgorithms = Array.isArray(config.selected_algorithms) ? config.selected_algorithms : [];
            
            // 调试配置信息
            console.log('会话配置:', {
                sessionId: sessionId,
                config: config,
                symbols: symbols,
                timeframes: timeframes,
                selectedAlgorithms: selectedAlgorithms
            });
            
            // 详细调试算法信息
            console.log('算法调试:', {
                rawConfig: session.config,
                selectedAlgorithms: selectedAlgorithms,
                selectedAlgorithmsLength: selectedAlgorithms.length,
                selectedAlgorithmsType: typeof selectedAlgorithms,
                selectedAlgorithmsIsArray: Array.isArray(selectedAlgorithms)
            });
            
            // 安全地获取结果信息
            const results = session.results || {};
            let factorsCount = 0;
            
            try {
                factorsCount = session.factors_count || results.total_factors || results.factors_info?.total_factors || 0;
                // 确保是数字
                factorsCount = parseInt(factorsCount) || 0;
            } catch (countError) {
                console.error('获取因子数量失败:', countError);
                factorsCount = 0;
            }
            
            console.log(`会话 ${index + 1} 因子数量:`, factorsCount);
            
            // 安全地构建HTML
            const safeTimeStr = timeStr.replace(/[<>]/g, '');
            const safeSymbols = symbols.map(s => String(s || '').replace(/[<>]/g, '')).join(', ') || '未知';
            const safeTimeframes = timeframes.map(t => String(t || '').replace(/[<>]/g, '')).join(', ') || '未知';
            const safeSelectedAlgorithms = selectedAlgorithms.map(a => String(a || '').replace(/[<>]/g, '')).join(', ') || '未知';
            const safeFactorsCount = String(factorsCount);
            const safeStatus = String(session.status || '未知').replace(/[<>]/g, '');
            
            row.innerHTML = `
                <td>${safeTimeStr}</td>
                <td>${safeSymbols}</td>
                <td>${safeTimeframes}</td>
                <td>${safeSelectedAlgorithms}</td>
                <td>${safeFactorsCount}</td>
                <td>
                    <span class="badge bg-${session.status === 'completed' ? 'success' : session.status === 'running' ? 'warning' : 'secondary'}">
                        ${session.status === 'completed' ? '完成' : session.status === 'running' ? '进行中' : '未知'}
                    </span>
                </td>
                <td>
                    ${session.status === 'completed' ? 
                        `<button class="btn btn-sm btn-outline-primary" onclick="viewMiningResult('${sessionId}')">
                            <i class="fas fa-eye me-1"></i>查看
                        </button>` : 
                        '<span class="text-muted">-</span>'
                    }
                </td>
            `;
            
            historyTableBody.appendChild(row);
            console.log(`第 ${index + 1} 行添加完成`);
            
        } catch (sessionError) {
            console.error(`处理会话 ${index + 1} 时出错:`, sessionError);
            // 添加错误行
            const errorRow = document.createElement('tr');
            errorRow.innerHTML = `
                <td colspan="7" class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>会话数据错误: ${sessionError.message}
                </td>
            `;
            historyTableBody.appendChild(errorRow);
        }
    });
    
    console.log(`挖掘历史更新完成，共 ${sessions.length} 条记录`);
}

/**
 * 查看挖掘结果
 */
async function viewMiningResult(sessionId) {
    try {
        console.log('查看挖掘结果:', sessionId);
        
        const response = await fetch(`/api/mining/result/${sessionId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // 获取响应文本
        const responseText = await response.text();
        console.log('挖掘结果响应文本:', responseText);
        
        // 清理NaN值，替换为null
        const cleanedText = responseText.replace(/: NaN/g, ': null');
        console.log('清理后的响应文本:', cleanedText);
        
        let data;
        try {
            // 尝试解析清理后的JSON
            data = JSON.parse(cleanedText);
        } catch (parseError) {
            console.error('JSON解析失败:', parseError);
            console.error('清理后的响应文本:', cleanedText);
            throw new Error(`JSON解析失败: ${parseError.message}`);
        }
        
        console.log('挖掘结果数据:', data);
        
        if (data.success !== false) {
            // 显示结果
            showMiningResults(data);
            
            // 滚动到结果区域
            const miningResults = document.getElementById('miningResults');
            if (miningResults) {
                miningResults.scrollIntoView({ behavior: 'smooth' });
            }
        } else {
            showAlert('error', data.error || '获取挖掘结果失败');
        }
    } catch (error) {
        console.error('获取挖掘结果失败:', error);
        showAlert('error', `获取挖掘结果失败: ${error.message}`);
    }
}

/**
 * 显示状态信息（替代弹窗）
 */
function showAlert(type, message) {
    console.log(`${type}: ${message}`);
    
    const statusDisplay = document.getElementById('statusDisplay');
    const statusMessage = document.getElementById('statusMessage');
    
    if (statusDisplay && statusMessage) {
        // 设置样式和消息
        statusDisplay.className = `alert alert-${type === 'error' ? 'danger' : type === 'warning' ? 'warning' : type === 'success' ? 'success' : 'info'} mb-3`;
        statusMessage.textContent = message;
        
        // 显示状态区域
        statusDisplay.style.display = 'block';
        
        // 自动隐藏成功和警告消息（5秒后）
        if (type === 'success' || type === 'warning') {
            setTimeout(() => {
                statusDisplay.style.display = 'none';
            }, 5000);
        }
        
        // 错误消息保持显示，直到用户手动关闭或新的状态
        if (type === 'error') {
            // 添加关闭按钮
            if (!statusDisplay.querySelector('.btn-close')) {
                const closeBtn = document.createElement('button');
                closeBtn.className = 'btn-close';
                closeBtn.setAttribute('type', 'button');
                closeBtn.setAttribute('aria-label', 'Close');
                closeBtn.onclick = () => {
                    statusDisplay.style.display = 'none';
                };
                statusDisplay.appendChild(closeBtn);
            }
        }
    }
}

/**
 * 更新子进度条
 */
function updateSubProgress(message, mainProgress) {
    console.log('更新子进度条:', message, mainProgress);
    
    // 显示子进度条容器
    const subProgressContainer = document.getElementById('subProgressContainer');
    if (subProgressContainer) {
        subProgressContainer.style.display = 'block';
    }
    
    // 解析消息，确定当前执行的因子类型
    let currentFactorType = null;
    let currentStatus = 'running';
    let progressPercent = 0;
    
    if (message.includes('机器学习因子') || message.includes('ML因子') || 
        message.includes('机器学习') || message.includes('集成模型') || 
        message.includes('PCA') || message.includes('特征选择') || 
        message.includes('滚动ML') || message.includes('自适应ML')) {
        currentFactorType = 'ml';
    } else if (message.includes('技术因子')) {
        currentFactorType = 'technical';
    } else if (message.includes('统计因子')) {
        currentFactorType = 'statistical';
    } else if (message.includes('高级因子')) {
        currentFactorType = 'advanced';
    }
    
    // 根据主进度计算子进度
    if (mainProgress <= 20) {
        progressPercent = (mainProgress / 20) * 100;
    } else if (mainProgress <= 40) {
        progressPercent = ((mainProgress - 20) / 20) * 100;
    } else if (mainProgress <= 60) {
        progressPercent = ((mainProgress - 40) / 20) * 100;
    } else if (mainProgress <= 80) {
        progressPercent = ((mainProgress - 60) / 20) * 100;
    } else {
        progressPercent = 100;
    }
    
    // 更新所有子进度条
    updateSubProgressItem('ml', currentFactorType === 'ml', progressPercent, message);
    updateSubProgressItem('technical', currentFactorType === 'technical', progressPercent, message);
    updateSubProgressItem('statistical', currentFactorType === 'statistical', progressPercent, message);
    updateSubProgressItem('advanced', currentFactorType === 'advanced', progressPercent, message);
}

/**
 * 更新单个子进度条项目
 */
function updateSubProgressItem(type, isActive, progress, message) {
    const container = document.getElementById(`${type}SubProgress`);
    if (!container) return;
    
    const progressBar = container.querySelector('.sub-progress-fill');
    const status = container.querySelector('.sub-progress-status');
    const details = container.querySelector('.sub-progress-details');
    
    if (isActive) {
        // 显示并激活当前因子类型
        container.style.display = 'block';
        
        // 更新进度条
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.classList.add('animate');
        }
        
        // 更新状态
        if (status) {
            status.textContent = '执行中...';
            status.className = 'sub-progress-status running';
        }
        
        // 更新详细信息
        if (details) {
            details.textContent = message || '正在处理...';
        }
    } else {
        // 隐藏非活跃的因子类型
        container.style.display = 'none';
    }
}

/**
 * 重置子进度条
 */
function resetSubProgress() {
    const subProgressContainer = document.getElementById('subProgressContainer');
    if (subProgressContainer) {
        subProgressContainer.style.display = 'none';
    }
    
    // 重置所有子进度条
    ['ml', 'technical', 'statistical', 'advanced'].forEach(type => {
        const container = document.getElementById(`${type}SubProgress`);
        if (container) {
            const progressBar = container.querySelector('.sub-progress-fill');
            const status = container.querySelector('.sub-progress-status');
            const details = container.querySelector('.sub-progress-details');
            
            if (progressBar) {
                progressBar.style.width = '0%';
                progressBar.classList.remove('animate');
            }
            
            if (status) {
                status.textContent = '准备中...';
                status.className = 'sub-progress-status';
            }
            
            if (details) {
                details.textContent = '初始化...';
            }
        }
    });
}

/**
 * 加载算法列表
 */
async function loadAlgorithms() {
    try {
        console.log('开始加载算法列表...');
        const response = await fetch('/api/mining/algorithms');
        const result = await response.json();
        
        if (result.success) {
            console.log(`成功加载 ${result.algorithms.length} 个算法`);
            renderAlgorithmSelector(result.algorithms);
        } else {
            console.error('加载算法失败:', result.error);
            showNotification('加载算法失败: ' + result.error, 'error');
        }
    } catch (error) {
        console.error('加载算法异常:', error);
        showNotification('加载算法异常: ' + error.message, 'error');
    }
}

/**
 * 渲染算法选择器
 */
function renderAlgorithmSelector(algorithms) {
    const container = document.getElementById('algorithm_selector');
    if (!container) {
        console.warn('算法选择器容器不存在');
        return;
    }
    
    // 按分类组织算法
    const algorithmsByCategory = {};
    algorithms.forEach(algo => {
        const category = algo.category || 'other';
        if (!algorithmsByCategory[category]) {
            algorithmsByCategory[category] = [];
        }
        algorithmsByCategory[category].push(algo);
    });
    
    // 清空容器
    container.innerHTML = '';
    
    // 创建分类标题和算法列表
    Object.keys(algorithmsByCategory).forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'algorithm-category mb-3';
        
        const categoryTitle = document.createElement('h6');
        categoryTitle.className = 'text-primary mb-2';
        categoryTitle.textContent = category.toUpperCase();
        categoryDiv.appendChild(categoryTitle);
        
        const algorithmList = document.createElement('div');
        algorithmList.className = 'algorithm-list';
        
        algorithmsByCategory[category].forEach(algo => {
            const algoDiv = document.createElement('div');
            algoDiv.className = 'form-check algorithm-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input algorithm-checkbox';
            checkbox.id = `algo_${algo.id}`;
            checkbox.value = algo.id;
            
            const label = document.createElement('label');
            label.className = 'form-check-label';
            label.htmlFor = `algo_${algo.id}`;
            label.innerHTML = `
                <strong>${algo.name}</strong>
                <small class="text-muted d-block">${algo.description || '无描述'}</small>
            `;
            
            algoDiv.appendChild(checkbox);
            algoDiv.appendChild(label);
            algorithmList.appendChild(algoDiv);
        });
        
        categoryDiv.appendChild(algorithmList);
        container.appendChild(categoryDiv);
    });
    
    console.log('算法选择器渲染完成');
}

/**
 * 获取选中的算法
 */
function getSelectedAlgorithms() {
    const checkboxes = document.querySelectorAll('.algorithm-checkbox:checked');
    return Array.from(checkboxes).map(checkbox => checkbox.value);
}

/**
 * 更新挖掘参数收集，包含算法选择
 */
function collectMiningParams() {
    const params = {
        symbols: getSelectedSymbols(),
        timeframes: getSelectedTimeframes(),
        selected_algorithms: getSelectedAlgorithms(),
        start_date: document.getElementById('start_date').value,
        end_date: document.getElementById('end_date').value,
        max_factors: parseInt(document.getElementById('max_factors').value) || 15,
        optimization_method: document.getElementById('optimization_method').value || 'greedy'
    };
    
    // 收集算法参数
    const selectedAlgos = getSelectedAlgorithms();
    selectedAlgos.forEach(algoId => {
        const algoParams = collectAlgorithmParams(algoId);
        if (Object.keys(algoParams).length > 0) {
            params[`${algoId}_params`] = algoParams;
        }
    });
    
    return params;
}

/**
 * 收集特定算法的参数（可扩展）
 */
function collectAlgorithmParams(algoId) {
    // 这里可以根据算法ID收集特定参数
    // 目前返回空对象，后续可以扩展
    return {};
}
