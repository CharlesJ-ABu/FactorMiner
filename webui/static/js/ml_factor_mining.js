/**
 * ML因子挖掘页面JavaScript
 */

// 全局变量
let localDataRows = [];
let miningSession = null;
let progressEventSource = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('ML因子挖掘页面初始化');
    initializePage();
});

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
    
    console.log('页面初始化完成');
}

/**
 * 绑定事件监听器
 */
function bindEventListeners() {
    // 表单提交
    const mlMiningForm = document.getElementById('mlMiningForm');
    if (mlMiningForm) {
        mlMiningForm.addEventListener('submit', handleMLMiningSubmit);
        console.log('ML挖掘表单事件监听器已绑定');
    } else {
        console.error('找不到ML挖掘表单元素');
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
    console.log('唯一交易对:', symbols);
    
    // 清空现有选项
    symbolsSelect.innerHTML = '<option value="">请选择...</option>';
    
    // 添加交易对选项
    symbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol;
        option.textContent = symbol;
        symbolsSelect.appendChild(option);
    });
    
    console.log(`交易对选择器更新完成，共 ${symbols.length} 个选项`);
}

/**
 * 更新时间框架选择器
 */
function updateTimeframesForSelection() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    const timeframesSelect = document.getElementById('timeframesSelect');
    
    if (!symbolsSelect || !timeframesSelect) return;
    
    const selectedSymbols = getSelectedValues('symbolsSelect');
    console.log('选中的交易对:', selectedSymbols);
    
    if (selectedSymbols.length === 0) {
        timeframesSelect.innerHTML = '<option value="">请先选择交易对</option>';
        return;
    }
    
    // 获取选中交易对可用的时间框架
    const availableTimeframes = new Set();
    selectedSymbols.forEach(symbol => {
        const symbolData = localDataRows.filter(row => row.symbol === symbol);
        symbolData.forEach(row => {
            if (row.timeframe) {
                availableTimeframes.add(row.timeframe);
            }
        });
    });
    
    const timeframes = Array.from(availableTimeframes).sort();
    console.log('可用时间框架:', timeframes);
    
    // 更新时间框架选择器
    timeframesSelect.innerHTML = '<option value="">请选择...</option>';
    timeframes.forEach(timeframe => {
        const option = document.createElement('option');
        option.value = timeframe;
        option.textContent = timeframe;
        timeframesSelect.appendChild(option);
    });
    
    console.log(`时间框架选择器更新完成，共 ${timeframes.length} 个选项`);
    
    // 更新时间范围
    updateRangeForSelection();
}

/**
 * 更新时间范围选择器
 */
function updateRangeForSelection() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    const timeframesSelect = document.getElementById('timeframesSelect');
    
    if (!symbolsSelect || !timeframesSelect) return;
    
    const selectedSymbols = getSelectedValues('symbolsSelect');
    const selectedTimeframes = getSelectedValues('timeframesSelect');
    
    if (selectedSymbols.length === 0 || selectedTimeframes.length === 0) {
        return;
    }
    
    // 获取选中交易对和时间框架的数据范围
    let minDate = null;
    let maxDate = null;
    
    selectedSymbols.forEach(symbol => {
        selectedTimeframes.forEach(timeframe => {
            const symbolData = localDataRows.filter(row => 
                row.symbol === symbol && row.timeframe === timeframe
            );
            
            symbolData.forEach(row => {
                if (row.start_date) {
                    const startDate = new Date(row.start_date);
                    if (!minDate || startDate < minDate) {
                        minDate = startDate;
                    }
                }
                if (row.end_date) {
                    const endDate = new Date(row.end_date);
                    if (!maxDate || endDate > maxDate) {
                        maxDate = endDate;
                    }
                }
            });
        });
    });
    
    // 更新日期输入框
    if (minDate && maxDate) {
        const startDateInput = document.getElementById('startDate');
        const endDateInput = document.getElementById('endDate');
        
        if (startDateInput) {
            startDateInput.value = minDate.toISOString().split('T')[0];
        }
        if (endDateInput) {
            endDateInput.value = maxDate.toISOString().split('T')[0];
        }
        
        console.log(`时间范围更新: ${minDate.toISOString().split('T')[0]} 到 ${maxDate.toISOString().split('T')[0]}`);
    }
}

/**
 * 获取选择器的选中值
 */
function getSelectedValues(selectId) {
    const select = document.getElementById(selectId);
    if (!select) return [];
    
    const selectedOptions = Array.from(select.selectedOptions);
    return selectedOptions.map(option => option.value).filter(value => value !== '');
}

/**
 * 处理ML挖掘表单提交
 */
async function handleMLMiningSubmit(event) {
    event.preventDefault();
    console.log('ML挖掘表单提交');
    
    try {
        // 获取表单数据
        const formData = new FormData(event.target);
        const miningData = {
            symbols: getSelectedValues('symbolsSelect'),
            timeframes: getSelectedValues('timeframesSelect'),
            start_date: formData.get('startDate'),
            end_date: formData.get('endDate'),
            window: parseInt(formData.get('window')) || 252,
            n_components: parseInt(formData.get('nComponents')) || 10,
            k_best: parseInt(formData.get('kBest')) || 20,
            rolling_window: parseInt(formData.get('rollingWindow')) || 252,
            adaptive_threshold: parseFloat(formData.get('adaptiveThreshold')) || 0.8,
            optimization_method: formData.get('optimizationMethod'),
            max_factors: parseInt(formData.get('maxFactors')) || 20,
            min_ic: parseFloat(formData.get('minIC')) || 0.015,
            min_ir: parseFloat(formData.get('minIR')) || 0.08,
            min_sample_size: parseInt(formData.get('minSampleSize')) || 50
        };
        
        console.log('ML挖掘配置:', miningData);
        
        // 验证数据
        if (miningData.symbols.length === 0) {
            showAlert('error', '请选择至少一个交易对');
            return;
        }
        
        if (miningData.timeframes.length === 0) {
            showAlert('error', '请选择至少一个时间框架');
            return;
        }
        
        // 更新按钮状态
        updateStartButton(true);
        
        // 显示等待状态
        showWaitingState();
        
        // 启动ML挖掘
        const response = await fetch('/api/mining/ml_mining', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(miningData)
        });
        
        const result = await response.json();
        console.log('ML挖掘启动结果:', result);
        
        if (result.success) {
            // 保存会话ID
            miningSession = result.session_id;
            
            // 显示预估时间信息
            if (result.estimated_time) {
                const estimatedTime = formatTime(result.estimated_time);
                showAlert('info', `ML挖掘已启动！预估总时间: ${estimatedTime}`);
                
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
            showAlert('error', result.error || 'ML挖掘启动失败');
        }
        
    } catch (error) {
        console.error('ML挖掘失败:', error);
        showAlert('error', `ML挖掘过程中发生错误: ${error.message}`);
    } finally {
        // 恢复按钮状态
        updateStartButton(false);
    }
}

/**
 * 更新开始按钮状态
 */
function updateStartButton(isLoading) {
    const startBtn = document.getElementById('startMiningBtn');
    if (!startBtn) return;
    
    if (isLoading) {
        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>启动中...';
    } else {
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-rocket me-2"></i>开始ML因子挖掘';
    }
}

/**
 * 显示等待状态
 */
function showWaitingState() {
    const waitingState = document.getElementById('waitingState');
    if (waitingState) {
        waitingState.style.display = 'block';
    }
    
    // 隐藏挖掘配置
    const miningConfig = document.getElementById('miningConfig');
    if (miningConfig) {
        miningConfig.style.display = 'none';
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
 * 启动进度监控
 */
function startProgressMonitoring(sessionId) {
    console.log('启动进度监控，会话ID:', sessionId);
    
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
            }
        } catch (error) {
            console.error('解析进度数据失败:', error);
        }
    };
    
    progressEventSource.onerror = function(error) {
        console.error('进度流错误:', error);
        // 如果SSE失败，回退到轮询
        fallbackToPolling(sessionId);
    };
}

/**
 * 回退到轮询
 */
function fallbackToPolling(sessionId) {
    console.log('回退到轮询模式');
    
    if (progressEventSource) {
        progressEventSource.close();
    }
    
    // 使用轮询获取进度
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/mining/status/${sessionId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    updateProgressDisplay(data);
                    
                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        handleMiningCompleted(sessionId);
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        handleMiningError(data.error);
                    }
                }
            }
        } catch (error) {
            console.error('轮询失败:', error);
        }
    }, 1000);
}

/**
 * 更新进度显示
 */
function updateProgressDisplay(data) {
    const { progress, current_step, messages, time_info, progress_info, system_info } = data;
    console.log('更新进度显示:', { progress, current_step, messages, time_info, progress_info, system_info });
    
    // 计算总体进度
    let overallProgress = 0;
    let completedSteps = 0;
    const totalSteps = Object.keys(progress).length;
    
    // 更新每个步骤的进度
    Object.keys(progress).forEach(stepKey => {
        const stepProgress = progress[stepKey];
        const frontendStepId = STEP_MAPPING[stepKey];
        
        if (!frontendStepId) {
            console.warn(`未知的步骤名称: ${stepKey}`);
            return;
        }
        
        const stepElement = document.getElementById(frontendStepId);
        console.log(`步骤 ${stepKey} -> ${frontendStepId}, 进度: ${stepProgress}%, 元素:`, stepElement);
        
        if (stepElement) {
            const progressBar = stepElement.querySelector('.progress-fill');
            const stepIcon = stepElement.querySelector('.step-icon');
            const stepDetails = stepElement.querySelector('.step-details');
            const stepTime = stepElement.querySelector('.step-time');
            
            // 更新进度条
            if (progressBar) {
                progressBar.style.width = `${stepProgress}%`;
                console.log(`更新进度条 ${frontendStepId}: ${stepProgress}%`);
            }
            
            // 更新图标和状态
            if (stepIcon) {
                if (stepProgress === 100) {
                    stepIcon.className = 'step-icon completed';
                    stepIcon.innerHTML = '<i class="fas fa-check"></i>';
                    completedSteps++;
                    console.log(`步骤 ${frontendStepId} 完成`);
                } else if (stepKey === current_step) {
                    stepIcon.className = 'step-icon running';
                    stepIcon.innerHTML = '<i class="fas fa-cog fa-spin"></i>';
                    console.log(`步骤 ${frontendStepId} 运行中`);
                }
            }
            
            // 更新详细信息
            if (stepDetails && messages && messages.length > 0) {
                const stepMessages = messages.filter(msg => msg.step === stepKey);
                if (stepMessages.length > 0) {
                    const lastMessage = stepMessages[stepMessages.length - 1];
                    stepDetails.textContent = lastMessage.message || `进度: ${stepProgress}%`;
                    console.log(`更新步骤详情 ${frontendStepId}:`, lastMessage.message);
                    
                    // 处理子进度条（针对因子构建步骤）
                    if (stepKey === 'factor_building') {
                        updateSubProgress(lastMessage.message, stepProgress);
                    }
                }
            }
            
            // 更新时间信息
            if (stepTime && progress_info && progress_info[stepKey]) {
                const stepInfo = progress_info[stepKey];
                if (stepKey === current_step && stepInfo.current_step_start) {
                    const elapsed = stepInfo.current_step_elapsed || 0;
                    const estimated = stepInfo.estimated_time || 0;
                    const remaining = stepInfo.current_step_remaining || 0;
                    
                    if (remaining > 0) {
                        stepTime.textContent = `已用: ${elapsed}s | 剩余: ${remaining}s`;
                    } else {
                        stepTime.textContent = `已用: ${elapsed}s | 预计: ${estimated}s`;
                    }
                } else if (stepProgress === 100) {
                    const estimated = stepInfo.estimated_time || 0;
                    stepTime.textContent = `完成 | 用时: ${estimated}s`;
                } else {
                    const estimated = stepInfo.estimated_time || 0;
                    stepTime.textContent = `预计: ${estimated}s`;
                }
            }
        } else {
            console.error(`找不到步骤元素: ${frontendStepId}`);
        }
    });
    
    // 更新总体进度
    if (totalSteps > 0) {
        overallProgress = Math.round((completedSteps / totalSteps) * 100);
    }
    
    updateOverallProgress(overallProgress, current_step, messages, time_info, system_info);
}

/**
 * 更新总体进度
 */
function updateOverallProgress(progress, currentStep, messages, timeInfo, systemInfo) {
    const overallProgress = document.getElementById('overallProgress');
    const overallDetails = document.getElementById('overallDetails');
    const timeDisplay = document.getElementById('timeDisplay');
    const systemDisplay = document.getElementById('systemDisplay');
    
    if (overallProgress) {
        overallProgress.style.width = `${progress}%`;
    }
    
    if (overallDetails) {
        if (currentStep) {
            const stepNames = {
                'data_loading': '数据加载',
                'factor_building': '因子构建',
                'factor_evaluation': '因子评估',
                'factor_optimization': '因子优化',
                'result_saving': '结果保存'
            };
            overallDetails.textContent = `当前步骤: ${stepNames[currentStep] || currentStep}`;
        } else {
            overallDetails.textContent = '准备开始...';
        }
    }
    
    if (timeInfo && timeDisplay) {
        const { total_estimated_time, elapsed_time, remaining_time } = timeInfo;
        if (remaining_time > 0) {
            timeDisplay.textContent = `已用: ${elapsed_time}s | 剩余: ${remaining_time}s`;
        } else if (total_estimated_time > 0) {
            timeDisplay.textContent = `预估总时间: ${formatTime(total_estimated_time)}`;
        }
    }
    
    if (systemInfo && systemDisplay) {
        const { cpu_count, memory_gb, memory_percent } = systemInfo;
        systemDisplay.textContent = `CPU: ${cpu_count}核 | 内存: ${memory_gb}GB (${memory_percent}%)`;
    }
}

/**
 * 处理挖掘完成
 */
async function handleMiningCompleted(sessionId) {
    console.log('挖掘完成，会话ID:', sessionId);
    
    try {
        // 获取挖掘结果
        const response = await fetch(`/api/mining/result/${sessionId}`);
        if (response.ok) {
            // 获取响应文本
            const responseText = await response.text();
            console.log('ML挖掘结果响应文本:', responseText);
            
            // 清理NaN值，替换为null
            const cleanedText = responseText.replace(/: NaN/g, ': null');
            console.log('清理后的ML挖掘结果文本:', cleanedText);
            
            let result;
            try {
                // 尝试解析清理后的JSON
                result = JSON.parse(cleanedText);
            } catch (parseError) {
                console.error('ML挖掘结果JSON解析失败:', parseError);
                console.error('清理后的响应文本:', cleanedText);
                throw new Error(`JSON解析失败: ${parseError.message}`);
            }
            
            if (result.success) {
                showMiningResults(result);
                showAlert('success', 'ML因子挖掘完成！');
            } else {
                showAlert('error', `获取结果失败: ${result.error}`);
            }
        } else {
            showAlert('error', `获取结果失败: ${response.status}`);
        }
    } catch (error) {
        console.error('获取挖掘结果失败:', error);
        showAlert('error', '获取挖掘结果失败');
    }
}

/**
 * 处理挖掘错误
 */
function handleMiningError(error) {
    console.error('挖掘错误:', error);
    showAlert('error', `挖掘失败: ${error}`);
}

/**
 * 显示挖掘结果
 */
function showMiningResults(result) {
    const miningResults = document.getElementById('miningResults');
    if (!miningResults) return;
    
    // 显示结果界面
    miningResults.style.display = 'block';
    
    // 更新结果信息
    const totalFactors = document.getElementById('totalFactors');
    const factorsCount = result.factors_count || result.factors_info?.total_factors || 0;
    
    if (totalFactors) {
        totalFactors.textContent = factorsCount;
    }
    
    // 可以添加更多结果展示逻辑
    console.log('显示挖掘结果:', result);
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
    
    if (message.includes('机器学习因子') || message.includes('ML因子')) {
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
 * 显示提示信息
 */
function showAlert(type, message) {
    const alertContainer = document.getElementById('statusAlerts');
    if (!alertContainer) return;
    
    // 隐藏所有提示
    const alerts = alertContainer.querySelectorAll('.alert');
    alerts.forEach(alert => alert.style.display = 'none');
    
    // 显示指定类型的提示
    const alertElement = document.getElementById(`${type}Alert`);
    if (alertElement) {
        const messageElement = alertElement.querySelector(`#${type}Message`);
        if (messageElement) {
            messageElement.textContent = message;
        }
        alertElement.style.display = 'block';
        
        // 自动隐藏
        setTimeout(() => {
            alertElement.style.display = 'none';
        }, 5000);
    }
}

/**
 * 格式化时间
 */
function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds}秒`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}分${remainingSeconds}秒`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const remainingMinutes = Math.floor((seconds % 3600) / 60);
        return `${hours}小时${remainingMinutes}分`;
    }
}

// 步骤名称映射（后端步骤名称 -> 前端步骤ID）
const STEP_MAPPING = {
    'data_loading': 'step1',
    'factor_building': 'step2', 
    'factor_evaluation': 'step3',
    'factor_optimization': 'step4',
    'result_saving': 'step5'
};
