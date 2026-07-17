/**
 * chat-render.js — 共享的对话消息渲染函数
 * 被 chat-viewer.html 和 thinking_dataset.html 共同引用。
 * 所有函数挂在全局作用域。
 */

/* ========= 基础工具 ========= */

function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _looksLikeHtml(text) {
    return /^\s*<!DOCTYPE\s+html|^\s*<html[\s>]/i.test(text);
}

function _looksLikeMarkdown(text) {
    return /^#{1,6}\s|^\s*[-*+]\s|\*\*|__|\[.*\]\(.*\)|```|^\s*>\s|^\|.*\|/m.test(text);
}

function renderKV(entries) {
    const wrap = document.createElement('div');
    wrap.className = 'input-kv';
    entries.forEach(function(pair) {
        const k = pair[0], v = pair[1];
        const row = document.createElement('div');
        row.className = 'kv-row';
        const keyEl = document.createElement('span');
        keyEl.className = 'kv-key';
        keyEl.textContent = k + ':';
        const valEl = document.createElement('span');
        valEl.className = 'kv-val';
        valEl.textContent = typeof v === 'string' ? v : JSON.stringify(v, null, 2);
        row.appendChild(keyEl);
        row.appendChild(valEl);
        wrap.appendChild(row);
    });
    return wrap;
}

function _wrapCollapsible(el) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;flex-direction:column;min-width:0;max-width:100%;';
    el.classList.add('collapsible', 'collapsed');
    wrapper.appendChild(el);
    const btn = document.createElement('button');
    btn.className = 'text-toggle-btn';
    btn.textContent = '展开全部';
    btn.onclick = function() {
        const isCollapsed = el.classList.toggle('collapsed');
        btn.textContent = isCollapsed ? '展开全部' : '收起';
    };
    wrapper.appendChild(btn);
    return wrapper;
}

/* ========= Block 渲染器 ========= */

function renderTextBlock(block) {
    const text = block.text || '';
    const el = document.createElement('div');
    el.className = 'block text';

    if (_looksLikeHtml(text)) {
        el.textContent = text.slice(0, 200) + (text.length > 200 ? '...' : '');
        const btn = document.createElement('a');
        btn.className = 'html-preview-btn';
        btn.textContent = '预览 HTML';
        btn.href = '#';
        btn.onclick = function(e) {
            e.preventDefault();
            const blob = new Blob([text], {type: 'text/html;charset=utf-8'});
            window.open(URL.createObjectURL(blob), '_blank');
        };
        el.appendChild(btn);
        return el;
    }

    if (typeof marked !== 'undefined' && _looksLikeMarkdown(text)) {
        el.classList.add('md-rendered');
        el.innerHTML = marked.parse(text);
        return text.length > 800 ? _wrapCollapsible(el) : el;
    }

    el.textContent = text;
    return text.length > 1500 ? _wrapCollapsible(el) : el;
}

function renderToolUseBlock(block) {
    const el = document.createElement('div');
    el.className = 'block tool-use';
    const hdr = document.createElement('div');
    hdr.className = 'block-hdr';
    hdr.innerHTML = '<span class="badge badge-tool-use">tool_use</span>' +
        '<span class="tool-name">' + esc(block.name || '') + '</span>' +
        '<span class="tool-id">' + esc(block.id || '') + '</span>';
    el.appendChild(hdr);

    const body = document.createElement('div');
    body.className = 'block-body';
    const input = block.input || {};
    if (input.command !== undefined) {
        const lbl = document.createElement('div');
        lbl.className = 'cmd-label';
        lbl.textContent = '$ command';
        const pre = document.createElement('pre');
        pre.className = 'cmd-pre';
        pre.textContent = input.command;
        body.appendChild(lbl);
        body.appendChild(pre);
        const others = Object.entries(input).filter(function(p) { return p[0] !== 'command'; });
        if (others.length) body.appendChild(renderKV(others));
    } else if (Object.keys(input).length) {
        const contentVal = typeof input.content === 'string' ? input.content : '';
        const hasHtml = contentVal && _looksLikeHtml(contentVal);
        if (hasHtml) {
            const nonContent = Object.entries(input).filter(function(p) { return p[0] !== 'content'; });
            if (nonContent.length) body.appendChild(renderKV(nonContent));
            const preview = document.createElement('div');
            preview.style.cssText = 'margin-top:6px;';
            const pre = document.createElement('pre');
            pre.style.cssText = 'max-height:120px;overflow:auto;font-size:11px;color:#6b7280;background:#f9fafb;padding:6px;border-radius:4px;white-space:pre-wrap;word-break:break-all;';
            pre.textContent = contentVal.slice(0, 500) + (contentVal.length > 500 ? '\n...' : '');
            preview.appendChild(pre);
            const btn = document.createElement('a');
            btn.className = 'html-preview-btn';
            btn.textContent = '预览 HTML';
            btn.href = '#';
            btn.onclick = function(e) {
                e.preventDefault();
                const blob = new Blob([contentVal], {type: 'text/html;charset=utf-8'});
                window.open(URL.createObjectURL(blob), '_blank');
            };
            preview.appendChild(btn);
            body.appendChild(preview);
        } else {
            body.appendChild(renderKV(Object.entries(input)));
        }
    } else {
        body.textContent = '(no input)';
    }
    el.appendChild(body);
    return el;
}

function renderToolResultBlock(block) {
    const isError = block.is_error;
    const el = document.createElement('div');
    el.className = 'block tool-result' + (isError ? ' is-error' : '');
    const hdr = document.createElement('div');
    hdr.className = 'block-hdr';
    hdr.innerHTML = '<span class="badge ' + (isError ? 'badge-error' : 'badge-tool-result') + '">' +
        (isError ? 'error' : 'tool_result') + '</span>' +
        '<span class="tool-id">' + esc(block.tool_use_id || '') + '</span>';
    el.appendChild(hdr);

    const body = document.createElement('div');
    body.className = 'block-body';
    const rawContent = block.content;
    const text = typeof rawContent === 'string' ? rawContent : JSON.stringify(rawContent, null, 2);
    const pre = document.createElement('pre');
    pre.className = 'result-pre' + (isError ? ' is-error' : '');
    pre.textContent = text || '(no output)';
    body.appendChild(pre);
    el.appendChild(body);
    return el;
}

function renderOaiToolCallBlock(block) {
    const tc = block.tool_call || {};
    const fn = tc.function || {};
    const el = document.createElement('div');
    el.className = 'block tool-use';
    const hdr = document.createElement('div');
    hdr.className = 'block-hdr';
    hdr.innerHTML = '<span class="badge badge-tool-use">tool_call</span>' +
        '<span class="tool-name">' + esc(fn.name || '') + '</span>' +
        '<span class="tool-id">' + esc(tc.id || '') + '</span>';
    el.appendChild(hdr);

    const body = document.createElement('div');
    body.className = 'block-body';
    let parsed = null;
    try { parsed = JSON.parse(fn.arguments); } catch(e) {}
    if (parsed && parsed.command !== undefined) {
        const lbl = document.createElement('div');
        lbl.className = 'cmd-label';
        lbl.textContent = '$ command';
        const pre = document.createElement('pre');
        pre.className = 'cmd-pre';
        pre.textContent = parsed.command;
        body.appendChild(lbl);
        body.appendChild(pre);
        const others = Object.entries(parsed).filter(function(p) { return p[0] !== 'command'; });
        if (others.length) body.appendChild(renderKV(others));
    } else if (parsed && typeof parsed === 'object') {
        const contentVal = typeof parsed.content === 'string' ? parsed.content : '';
        const hasHtml = contentVal && _looksLikeHtml(contentVal);
        if (hasHtml) {
            const nonContent = Object.entries(parsed).filter(function(p) { return p[0] !== 'content'; });
            if (nonContent.length) body.appendChild(renderKV(nonContent));
            const preview = document.createElement('div');
            preview.style.cssText = 'margin-top:6px;';
            const pre = document.createElement('pre');
            pre.style.cssText = 'max-height:120px;overflow:auto;font-size:11px;color:#6b7280;background:#f9fafb;padding:6px;border-radius:4px;white-space:pre-wrap;word-break:break-all;';
            pre.textContent = contentVal.slice(0, 500) + (contentVal.length > 500 ? '\n...' : '');
            preview.appendChild(pre);
            const btn = document.createElement('a');
            btn.className = 'html-preview-btn';
            btn.textContent = '预览 HTML';
            btn.href = '#';
            btn.onclick = function(e) {
                e.preventDefault();
                const blob = new Blob([contentVal], {type: 'text/html;charset=utf-8'});
                window.open(URL.createObjectURL(blob), '_blank');
            };
            preview.appendChild(btn);
            body.appendChild(preview);
        } else {
            body.appendChild(renderKV(Object.entries(parsed)));
        }
    } else {
        let args = fn.arguments || '';
        try { args = JSON.stringify(JSON.parse(args), null, 2); } catch(e) {}
        const pre = document.createElement('pre');
        pre.className = 'cmd-pre';
        pre.textContent = args;
        body.appendChild(pre);
    }
    el.appendChild(body);
    return el;
}

function renderOaiToolResultBlock(block) {
    const el = document.createElement('div');
    el.className = 'block tool-role';
    const hdr = document.createElement('div');
    hdr.className = 'block-hdr tool-role-hdr';
    hdr.innerHTML = '<span class="badge badge-tool-role">tool_result</span>' +
        '<span class="tool-id">' + esc(block.tool_call_id || '') + '</span>';
    el.appendChild(hdr);
    const body = document.createElement('div');
    body.className = 'block-body';
    const pre = document.createElement('pre');
    pre.className = 'result-pre';
    pre.textContent = typeof block.content === 'string' ? block.content : JSON.stringify(block.content, null, 2);
    body.appendChild(pre);
    el.appendChild(body);
    return el;
}

function renderReasoningBlock(text, signature) {
    const el = document.createElement('div');
    el.className = 'block reasoning';
    const charCount = (text || '').length;
    let sigHtml = '';
    if (signature) {
        const sigLen = signature.length;
        sigHtml = '<div class="signature-row" onclick="this.nextElementSibling.classList.toggle(\'collapsed\'); this.querySelector(\'.signature-chevron\').classList.toggle(\'collapsed\')">' +
            '<span class="signature-chevron collapsed">&#9660;</span>' +
            '<span>🔏 signature (' + sigLen + ' chars)</span>' +
            '</div>' +
            '<div class="signature-body collapsed">' + esc(signature) + '</div>';
    }
    el.innerHTML = sigHtml +
        '<div class="reasoning-summary" onclick="this.nextElementSibling.classList.toggle(\'collapsed\'); this.querySelector(\'.reasoning-chevron\').classList.toggle(\'collapsed\')">' +
        '<span class="reasoning-chevron collapsed">&#9660;</span>' +
        '<span>思考过程 (' + charCount + ' 字)</span>' +
        '</div>' +
        '<div class="reasoning-body collapsed">' + esc(text || '') + '</div>';
    return el;
}

function renderSystemPrompt(content) {
    const text = typeof content === 'string' ? content
        : Array.isArray(content) ? content.map(function(b) { return b.text || ''; }).join('\n')
        : JSON.stringify(content, null, 2);
    const el = document.createElement('div');
    el.className = 'block system-prompt';
    const charCount = text.length;
    el.innerHTML = '<div class="system-summary" onclick="this.nextElementSibling.classList.toggle(\'collapsed\'); this.querySelector(\'.system-chevron\').classList.toggle(\'collapsed\')">' +
        '<span class="system-chevron collapsed">&#9660;</span>' +
        '<span>System Prompt (' + charCount + ' 字)</span>' +
        '</div>' +
        '<div class="system-body collapsed">' + esc(text) + '</div>';
    return el;
}

function renderFallbackBlock(block) {
    const el = document.createElement('div');
    el.className = 'block text';
    el.textContent = JSON.stringify(block, null, 2);
    return el;
}

/**
 * renderBlock — 分发器。
 * opts.onThinkingBlock: 可选钩子，覆盖 thinking block 渲染。
 */
function renderBlock(block, opts) {
    opts = opts || {};
    switch (block.type) {
        case 'text':            return renderTextBlock(block);
        case 'thinking':
            if (opts.onThinkingBlock) return opts.onThinkingBlock(block);
            return renderReasoningBlock(block.thinking, block.signature);
        case 'tool_use':        return renderToolUseBlock(block);
        case 'tool_result':     return renderToolResultBlock(block);
        case '_oai_tool_call':  return renderOaiToolCallBlock(block);
        case '_oai_tool_result':return renderOaiToolResultBlock(block);
        default:                return renderFallbackBlock(block);
    }
}

/* ========= Tool mode 逻辑 ========= */

function collectToolNames(container) {
    const names = new Set();
    container.querySelectorAll('.tool-row').forEach(function(el) {
        (el.dataset.toolNames || '').split(',').filter(Boolean).forEach(function(n) {
            if (n !== 'tool_result' && n !== 'tool') names.add(n);
        });
    });
    return Array.from(names).sort();
}

function buildFilterChips(chipContainer, panelContainer, state) {
    chipContainer.innerHTML = '';
    const allNames = collectToolNames(panelContainer);
    state.activeFilters = new Set(allNames);
    allNames.forEach(function(name) {
        const chip = document.createElement('span');
        chip.className = 'tool-filter-chip active';
        chip.textContent = name;
        chip.addEventListener('click', function() {
            if (state.activeFilters.has(name)) {
                state.activeFilters.delete(name);
                chip.classList.remove('active');
            } else {
                state.activeFilters.add(name);
                chip.classList.add('active');
            }
            applyToolMode(panelContainer, state);
        });
        chipContainer.appendChild(chip);
    });
}

function applyToolMode(container, state) {
    container.querySelectorAll('.tool-badge-row').forEach(function(el) { el.remove(); });

    if (state.mode === 0) {
        container.querySelectorAll('.tool-row').forEach(function(el) { el.style.display = ''; });
        return;
    }
    if (state.mode === 1) {
        container.querySelectorAll('.tool-row').forEach(function(el) {
            const names = (el.dataset.toolNames || '').split(',').filter(Boolean);
            const match = names.some(function(n) { return state.activeFilters.has(n); });
            el.style.display = match ? '' : 'none';
        });
        return;
    }
    // Mode 2: simplified
    const toolRows = container.querySelectorAll('.tool-row');
    toolRows.forEach(function(el) { el.style.display = 'none'; });

    const children = Array.from(container.children);
    let i = 0;
    while (i < children.length) {
        const el = children[i];
        if (!el.classList || !el.classList.contains('tool-row')) { i++; continue; }
        const counts = {};
        let lastToolEl = el;
        while (i < children.length && children[i].classList && children[i].classList.contains('tool-row')) {
            const names = (children[i].dataset.toolNames || '').split(',').filter(Boolean);
            names.forEach(function(n) { counts[n] = (counts[n] || 0) + 1; });
            lastToolEl = children[i];
            i++;
        }
        const parts = Object.entries(counts)
            .filter(function(p) { return p[0] !== 'tool' && p[0] !== 'tool_result'; })
            .map(function(p) { return '<span class="tool-badge-chip">' + esc(p[0]) + (p[1] > 1 ? ' x' + p[1] : '') + '</span>'; });
        if (!parts.length) continue;
        const badge = document.createElement('div');
        badge.className = 'tool-badge-row';
        badge.innerHTML = parts.join('');
        lastToolEl.after(badge);
    }
}

/* ========= 主渲染函数 ========= */

/**
 * renderChatMessages — 核心消息渲染循环。
 *
 * @param {HTMLElement} container — 渲染目标容器
 * @param {Array} messages — messages 数组
 * @param {Object} [opts] — 可选配置
 *   opts.onThinkingBlock(block) — 覆盖 thinking block 渲染
 *   opts.showToolMode — 是否显示 tool 模式切换栏 (default: false)
 *   opts.toolModeState — 外部传入的 tool mode 状态对象 {mode: 0, activeFilters: Set}
 *
 * @returns {Object} 返回 {toolModeState} 供外部控制
 */
function renderChatMessages(container, messages, opts) {
    opts = opts || {};

    if (!messages || !messages.length) {
        container.innerHTML += '<div style="padding:40px;text-align:center;color:var(--muted-foreground)">无消息内容</div>';
        return {};
    }

    var toolModeState = opts.toolModeState || { mode: 0, activeFilters: new Set() };
    var filterChipsContainer = null;

    if (opts.showToolMode) {
        var modeBar = document.createElement('div');
        modeBar.className = 'tool-mode-bar';
        var modeLabels = ['全部显示', '按工具筛选', '简化显示'];
        var modeBtns = [];
        modeLabels.forEach(function(label, i) {
            var btn = document.createElement('button');
            btn.className = 'mode-btn' + (i === toolModeState.mode ? ' active' : '');
            btn.textContent = label;
            btn.addEventListener('click', function() {
                toolModeState.mode = i;
                modeBtns.forEach(function(b, j) { b.classList.toggle('active', j === i); });
                if (filterChipsContainer) filterChipsContainer.style.display = i === 1 ? '' : 'none';
                applyToolMode(container, toolModeState);
            });
            modeBar.appendChild(btn);
            modeBtns.push(btn);
        });
        container.appendChild(modeBar);

        filterChipsContainer = document.createElement('div');
        filterChipsContainer.className = 'tool-filter-chips';
        filterChipsContainer.style.display = toolModeState.mode === 1 ? '' : 'none';
        container.appendChild(filterChipsContainer);
    }

    var blockOpts = {};
    if (opts.onThinkingBlock) blockOpts.onThinkingBlock = opts.onThinkingBlock;

    messages.forEach(function(msg, mi) {
        var role = msg.role || 'unknown';

        if (role === 'system') {
            container.appendChild(renderSystemPrompt(msg.content));
            return;
        }

        var blocks = [];
        var isToolRow = false;
        var toolRowType = '';

        if (role === 'tool') {
            blocks = [{ type: '_oai_tool_result', tool_call_id: msg.tool_call_id, content: msg.content }];
            isToolRow = true;
            toolRowType = 'result';
        } else if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
            if (Array.isArray(msg.content)) {
                blocks = blocks.concat(msg.content);
            } else if (msg.content) {
                blocks.push({ type: 'text', text: msg.content });
            }
            msg.tool_calls.forEach(function(tc) { blocks.push({ type: '_oai_tool_call', tool_call: tc }); });
            isToolRow = true;
            toolRowType = 'use';
        } else if (Array.isArray(msg.content)) {
            blocks = msg.content;
            var hasUse = blocks.some(function(b) { return b.type === 'tool_use' || b.type === '_oai_tool_call'; });
            var hasResult = blocks.some(function(b) { return b.type === 'tool_result' || b.type === '_oai_tool_result'; });
            if (hasUse || hasResult) {
                isToolRow = true;
                toolRowType = hasUse ? 'use' : 'result';
            }
        } else {
            blocks = [{ type: 'text', text: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2) }];
        }

        if (msg.reasoning_content) {
            blocks = [{ type: 'thinking', thinking: msg.reasoning_content }].concat(blocks);
        }

        var displayRole = role === 'tool' ? 'user' : role;

        var textBlocks = [];
        var toolBlocks = [];
        if (isToolRow && toolRowType === 'use') {
            blocks.forEach(function(b) {
                var bt = b.type || '';
                if (bt === 'tool_use' || bt === '_oai_tool_call') {
                    toolBlocks.push(b);
                } else {
                    textBlocks.push(b);
                }
            });
        }

        if (textBlocks.length > 0) {
            var textRow = document.createElement('div');
            textRow.className = 'msg-row ' + displayRole;
            var avatar1 = document.createElement('div');
            avatar1.className = 'avatar ' + displayRole;
            avatar1.textContent = displayRole === 'user' ? '👤' : '🤖';
            var col1 = document.createElement('div');
            col1.className = 'msg-bubble-col';
            if (msg._from_res) {
                var tag = document.createElement('div');
                tag.style.cssText = 'font-size:10px;color:#6366f1;font-weight:600;margin-bottom:4px;';
                tag.textContent = '↓ 来自 res 文件的回复';
                col1.appendChild(tag);
            }
            textBlocks.forEach(function(b) { var el = renderBlock(b, blockOpts); if (el) col1.appendChild(el); });
            if (displayRole === 'user') {
                textRow.appendChild(avatar1); textRow.appendChild(col1);
            } else {
                textRow.appendChild(col1); textRow.appendChild(avatar1);
            }
            container.appendChild(textRow);
        }

        var row = document.createElement('div');
        row.className = 'msg-row ' + displayRole;
        row.id = 'msg-' + mi;
        if (isToolRow) {
            row.classList.add('tool-row');
            if (toolRowType === 'use') row.classList.add('tool-use-row');
            if (toolRowType === 'result') row.classList.add('tool-result-row');
            var names = [];
            var renderBlks = toolBlocks.length > 0 ? toolBlocks : blocks;
            renderBlks.forEach(function(b) {
                if (b.type === 'tool_use') names.push(b.name || 'unknown');
                else if (b.type === '_oai_tool_call') {
                    var tcFn = (b.tool_call || {}).function || {};
                    names.push(tcFn.name || 'unknown');
                }
                else if (b.type === 'tool_result' || b.type === '_oai_tool_result') names.push('tool_result');
            });
            row.dataset.toolNames = names.join(',');
        }

        if (textBlocks.length > 0 && toolBlocks.length > 0) {
            var avatar = document.createElement('div');
            avatar.className = 'avatar ' + displayRole;
            avatar.textContent = '🤖';
            var col = document.createElement('div');
            col.className = 'msg-bubble-col';
            toolBlocks.forEach(function(b) { var el = renderBlock(b, blockOpts); if (el) col.appendChild(el); });
            row.appendChild(col); row.appendChild(avatar);
            container.appendChild(row);
        } else if (textBlocks.length === 0) {
            var avatar2 = document.createElement('div');
            avatar2.className = 'avatar ' + displayRole;
            avatar2.textContent = displayRole === 'user' ? '👤' : displayRole === 'assistant' ? '🤖' : '⚙️';
            var col2 = document.createElement('div');
            col2.className = 'msg-bubble-col';
            if (msg._from_res && !isToolRow) {
                var tag2 = document.createElement('div');
                tag2.style.cssText = 'font-size:10px;color:#6366f1;font-weight:600;margin-bottom:4px;';
                tag2.textContent = '↓ 来自 res 文件的回复';
                col2.appendChild(tag2);
            }
            blocks.forEach(function(b) { var el = renderBlock(b, blockOpts); if (el) col2.appendChild(el); });
            if (displayRole === 'user') {
                row.appendChild(avatar2); row.appendChild(col2);
            } else {
                row.appendChild(col2); row.appendChild(avatar2);
            }
            container.appendChild(row);
        }
    });

    if (opts.showToolMode && filterChipsContainer) {
        buildFilterChips(filterChipsContainer, container, toolModeState);
        applyToolMode(container, toolModeState);
    }

    return { toolModeState: toolModeState };
}
