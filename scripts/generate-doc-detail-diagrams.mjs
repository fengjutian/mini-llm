#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docsDir = path.join(root, 'docs');
const outDir = path.join(docsDir, 'assets', 'diagrams');

const specs = [
  ['TENSOR_ENGINE.md', 'Memory Layout', 'tensor-memory-layout.svg', 'Tensor 的二维索引如何通过 Stride 映射到连续内存。', 'tensor'],
  ['AUTOGRAD_ENGINE.md', '什么是计算图', 'autograd-computation-graph.svg', '前向计算图与反向梯度传播方向。', 'autograd'],
  ['TRANSFORMER_ENGINE.md', 'Self-Attention：', 'transformer-attention-heatmap.svg', '注意力矩阵将每个 Query 与所有 Key 的相关性表示为权重。', 'attention'],
  ['TRANSFORMER_BLOCK.md', 'Transformer Block 的整体架构', 'transformer-block-residual.svg', 'Pre-Norm Transformer Block 中的两条残差路径。', 'residual'],
  ['TRAINING_ENGINE.md', 'LLM 训练的核心循环', 'training-loss-loop.svg', '批次训练循环与随 Step 下降的示意 Loss 曲线。', 'training'],
  ['EVALUATION_ENGINE.md', 'Training Loss 与 Validation Loss', 'evaluation-overfitting.svg', '训练集与验证集曲线分离时出现过拟合。', 'evaluation'],
  ['REPRESENTATION_ENGINE.md', 'Semantic Geometry', 'representation-semantic-space.svg', '语义相近概念在向量空间中形成局部簇与关系方向。', 'scatter'],
  ['MEMORY_ARCHITECTURE.md', 'Short-term Memory 与 Long-term Memory', 'memory-hierarchy.svg', '工作记忆、情景记忆、语义记忆与长期记忆之间的写入和召回。', 'memory'],
  ['CAUSAL_REASONING_ENGINE.md', 'Causal Graph：', 'causal-intervention.svg', '观察、干预与反事实对应的三层因果推理。', 'causal'],
  ['PLANNING_ENGINE.md', 'State-space Search', 'planning-search-tree.svg', '状态空间搜索中的分支、剪枝与最优计划路径。', 'tree'],
  ['AGENT_ENGINE_RUNTIME.md', 'Agent Loop：', 'agent-loop.svg', 'Agent 在观察、思考、规划、行动和评估之间持续循环。', 'loop'],
  ['RUNTIME_ENGINE.md', 'Async Runtime：', 'runtime-async-timeline.svg', '异步调度让模型、工具和 Agent 任务重叠执行。', 'timeline'],
  ['OBSERVABILITY_ENGINE.md', 'Distributed Tracing', 'observability-trace.svg', '一次 Agent 请求跨越模型、工具和存储的 Trace 与 Span。', 'trace'],
  ['SECURITY_ENGINE.md', 'Trust Boundary', 'security-trust-boundaries.svg', '身份、策略与 Sandbox 组成逐层收紧的信任边界。', 'security'],
  ['MULTI_AGENT_ENGINE.md', 'Agent Orchestration', 'multi-agent-topology.svg', '协调器、专家 Agent 与共享记忆组成的协作拓扑。', 'network'],
  ['EMBODIED_AI_ENGINE.md', 'Sensor Fusion：', 'embodied-sensor-fusion.svg', '多模态传感器经时间同步和融合形成统一世界状态。', 'fusion'],
];

const colors = { accent: '#e45d32', blue: '#3178a5', green: '#4f8b62', gold: '#bd852b', ink: '#172126', muted: '#69777d', line: '#c4bdb1', paper: '#f7f4ed', panel: '#fffdf8' };
const esc = (s) => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const box = (x, y, w, h, label, tone = 'normal') => `<rect class="box ${tone}" x="${x}" y="${y}" width="${w}" height="${h}" rx="16"/><text class="label" x="${x + w / 2}" y="${y + h / 2 + 7}" text-anchor="middle">${esc(label)}</text>`;
const arrow = (x1, y1, x2, y2, cls = 'edge') => `<path class="${cls}" d="M${x1} ${y1} L${x2} ${y2}" marker-end="url(#arrow)"/>`;
const line = (x1, y1, x2, y2, cls = 'edge') => `<path class="${cls}" d="M${x1} ${y1} L${x2} ${y2}"/>`;

function frame(title, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="650" viewBox="0 0 1400 650" role="img" aria-labelledby="title desc">
  <title id="title">${esc(title)}</title><desc id="desc">mini-llm 技术机制图</desc>
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="currentColor"/></marker></defs>
  <style>
    :root{color-scheme:light dark}.bg{fill:${colors.paper}}.title{fill:${colors.ink};font:700 30px Inter,"Noto Sans SC","Microsoft YaHei",sans-serif}.kicker{fill:${colors.accent};font:700 16px ui-monospace,Consolas,monospace;letter-spacing:2px}.box{fill:${colors.panel};stroke:${colors.line};stroke-width:2}.box.hot{fill:#fff0e8;stroke:${colors.accent}}.box.cool{fill:#eaf3f7;stroke:${colors.blue}}.box.good{fill:#eaf4ec;stroke:${colors.green}}.label{fill:${colors.ink};font:650 18px Inter,"Noto Sans SC","Microsoft YaHei",sans-serif}.small{fill:${colors.muted};font:500 15px Inter,"Noto Sans SC","Microsoft YaHei",sans-serif}.edge{color:${colors.muted};fill:none;stroke:currentColor;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.accent-edge{color:${colors.accent};fill:none;stroke:currentColor;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}.axis{fill:none;stroke:${colors.muted};stroke-width:2}.grid{stroke:${colors.line};stroke-width:1;opacity:.45}.dot{stroke:${colors.panel};stroke-width:3}.area{opacity:.14}
    @media(prefers-color-scheme:dark){.bg{fill:#101416}.title,.label{fill:#edf2f3}.kicker{fill:#ef855d}.box{fill:#182024;stroke:#526168}.box.hot{fill:#32211c;stroke:#e17750}.box.cool{fill:#172934;stroke:#4d91bc}.box.good{fill:#192b20;stroke:#65a779}.small{fill:#a8b4b9}.grid{stroke:#3c484e}.dot{stroke:#101416}}
  </style><rect class="bg" width="1400" height="650" rx="24"/><text class="kicker" x="64" y="58">MINI-LLM · MECHANISM</text><text class="title" x="64" y="105">${esc(title)}</text>${body}</svg>`;
}

function render(kind) {
  if (kind === 'tensor') {
    const cells = Array.from({ length: 6 }, (_, i) => { const r = Math.floor(i / 3), c = i % 3; return `<rect x="${100 + c * 105}" y="${200 + r * 95}" width="90" height="78" rx="10" class="box ${i === 5 ? 'hot' : ''}"/><text class="label" x="${145 + c * 105}" y="${247 + r * 95}" text-anchor="middle">[${r},${c}]</text>`; }).join('');
    const memory = Array.from({ length: 6 }, (_, i) => `<rect x="${720 + i * 92}" y="246" width="78" height="76" rx="8" class="box ${i === 5 ? 'hot' : ''}"/><text class="label" x="${759 + i * 92}" y="292" text-anchor="middle">${i}</text>`).join('');
    return frame('Shape、Stride 与连续内存', `${cells}<text class="small" x="100" y="414">Shape = [2, 3] · Stride = [3, 1]</text>${arrow(430,285,680,285,'accent-edge')}${memory}<text class="small" x="720" y="370">offset = row × 3 + col × 1</text><text class="small" x="100" y="168">逻辑索引</text><text class="small" x="720" y="215">物理 Storage</text>`);
  }
  if (kind === 'autograd') return frame('计算图与反向传播', `${box(80,250,150,72,'x')}${box(80,370,150,72,'w','cool')}${box(340,290,180,82,'y = x · w')}${box(625,290,180,82,'Loss','hot')}${box(925,210,210,72,'∂L / ∂y','good')}${box(925,350,210,72,'∂L / ∂w','good')}${arrow(230,286,340,320)}${arrow(230,406,340,345)}${arrow(520,331,625,331)}${arrow(925,246,805,315,'accent-edge')}${arrow(925,386,520,355,'accent-edge')}<text class="small" x="80" y="190">前向：从输入计算 Loss</text><text class="small" x="925" y="180">反向：沿图累积梯度</text>`);
  if (kind === 'attention') {
    const vals=[[.9,.1,.2,.1],[.2,.8,.3,.1],[.1,.2,.85,.4],[.15,.1,.3,.9]]; const cells=vals.flatMap((row,r)=>row.map((v,c)=>`<rect x="${410+c*105}" y="165" width="92" height="92" transform="translate(0 ${r*92})" rx="7" fill="${colors.accent}" opacity="${v}"/><text class="label" x="${456+c*105}" y="220" transform="translate(0 ${r*92})" text-anchor="middle">${v.toFixed(2)}</text>`)).join('');
    return frame('Self-Attention 权重矩阵', `${cells}<text class="small" x="270" y="355" transform="rotate(-90 270 355)">Query Token</text><text class="small" x="585" y="590">Key Token</text><text class="small" x="930" y="250">每一行经 Softmax 后和为 1</text><text class="small" x="930" y="290">颜色越深，当前 Query 越关注该 Key</text>${arrow(900,310,830,340,'accent-edge')}`);
  }
  if (kind === 'residual') return frame('Pre-Norm Transformer Block', `${box(80,280,130,70,'Input','hot')}${box(310,210,170,70,'RMSNorm')}${box(560,210,170,70,'Attention','cool')}${box(810,210,140,70,'Add','hot')}${box(310,410,170,70,'RMSNorm')}${box(560,410,170,70,'FFN','cool')}${box(810,410,140,70,'Add','hot')}${box(1080,330,160,70,'Output','good')}${arrow(210,315,310,245)}${arrow(480,245,560,245)}${arrow(730,245,810,245)}${arrow(950,245,1080,365)}${arrow(210,315,810,245,'accent-edge')}${arrow(950,245,310,445)}${arrow(480,445,560,445)}${arrow(730,445,810,445)}${arrow(950,445,1080,365)}${arrow(950,245,810,445,'accent-edge')}<text class="small" x="420" y="185">Attention 子层</text><text class="small" x="430" y="535">FFN 子层</text>`);
  if (kind === 'training') return frame('训练循环与 Loss 收敛', `${box(70,210,160,70,'Batch')}${box(280,210,160,70,'Forward','cool')}${box(490,210,160,70,'Loss','hot')}${box(700,210,160,70,'Backward')}${box(910,210,160,70,'AdamW','good')}${arrow(230,245,280,245)}${arrow(440,245,490,245)}${arrow(650,245,700,245)}${arrow(860,245,910,245)}${arrow(990,280,150,360,'accent-edge')}<path class="axis" d="M760 530 H1290 M760 530 V340"/><path class="accent-edge" d="M780 370 C900 390 960 455 1080 470 S1220 493 1270 500"/><text class="small" x="760" y="325">Loss</text><text class="small" x="1240" y="565">Step</text><text class="small" x="70" y="400">下一批数据</text>`);
  if (kind === 'evaluation') return frame('训练曲线与过拟合拐点', `<path class="axis" d="M160 530 H1240 M160 530 V160"/><path class="accent-edge" d="M180 210 C350 300 520 380 760 445 S1050 480 1210 492"/><path d="M180 230 C370 315 540 390 720 420 S930 390 1210 260" fill="none" stroke="${colors.blue}" stroke-width="4"/><line x1="760" y1="160" x2="760" y2="530" stroke="${colors.gold}" stroke-width="3" stroke-dasharray="10 8"/><text class="label" x="945" y="475">Training Loss</text><text class="label" x="990" y="245">Validation Loss</text><text class="label" x="780" y="190">最佳 Checkpoint</text><text class="small" x="1015" y="320">泛化开始变差</text><text class="small" x="1050" y="565">Training Step</text>`);
  if (kind === 'scatter') {
    const dots=[[250,260,'猫'],[320,220,'狗'],[285,330,'宠物'],[630,240,'苹果'],[700,290,'香蕉'],[665,350,'水果'],[1010,230,'汽车'],[1080,300,'火车'],[1000,360,'交通']];
    return frame('语义空间中的局部几何', `<path class="axis" d="M120 535 H1270 M120 535 V155"/>${dots.map(([x,y,t],i)=>`<circle class="dot" cx="${x}" cy="${y}" r="12" fill="${i<3?colors.accent:i<6?colors.green:colors.blue}"/><text class="small" x="${x+18}" y="${y+5}">${t}</text>`).join('')}<path d="M300 390 C260 355 220 250 300 185 C375 150 410 300 350 380 Z" fill="${colors.accent}" class="area"/><path d="M650 400 C570 330 600 190 710 185 C790 260 755 380 650 400 Z" fill="${colors.green}" class="area"/><path d="M1010 410 C930 330 950 180 1090 190 C1160 290 1120 390 1010 410 Z" fill="${colors.blue}" class="area"/><text class="small" x="1090" y="570">语义方向 →</text>`);
  }
  if (kind === 'memory') return frame('记忆层级与巩固过程', `${box(80,250,210,80,'Working Memory','hot')}${box(410,170,220,80,'Episodic Memory','cool')}${box(410,350,220,80,'Semantic Memory','cool')}${box(790,250,220,80,'Consolidation')}${box(1120,250,210,80,'Long-term Store','good')}${arrow(290,290,410,210)}${arrow(290,290,410,390)}${arrow(630,210,790,280)}${arrow(630,390,790,300)}${arrow(1010,290,1120,290,'accent-edge')}${arrow(1120,320,300,330)}<text class="small" x="650" y="150">经历</text><text class="small" x="650" y="450">抽象知识</text><text class="small" x="640" y="365">Recall / Retrieval</text>`);
  if (kind === 'causal') return frame('观察、干预与反事实', `${box(90,180,260,80,'观察 P(Y | X)','cool')}${box(570,180,260,80,'干预 P(Y | do(X))','hot')}${box(1050,180,260,80,'反事实 Yₓ(u)','good')}${arrow(350,220,570,220)}${arrow(830,220,1050,220)}<circle cx="260" cy="440" r="46" class="box"/><circle cx="700" cy="440" r="46" class="box hot"/><circle cx="1140" cy="440" r="46" class="box good"/><text class="label" x="260" y="447" text-anchor="middle">相关</text><text class="label" x="700" y="447" text-anchor="middle">行动</text><text class="label" x="1140" y="447" text-anchor="middle">如果</text>${arrow(306,440,654,440)}${arrow(746,440,1094,440)}<text class="small" x="590" y="530">从“看见什么”走向“改变什么”和“若当时不同会怎样”</text>`);
  if (kind === 'tree') return frame('状态空间搜索与剪枝', `${box(600,155,180,64,'S₀','hot')}${box(250,290,170,64,'S₁')}${box(615,290,170,64,'S₂','good')}${box(980,290,170,64,'S₃')}${box(120,440,150,60,'S₄')}${box(360,440,150,60,'S₅')}${box(600,440,150,60,'Goal','good')}${box(850,440,150,60,'Pruned')}${box(1090,440,150,60,'S₈')}${line(650,219,335,290)}${line(690,219,700,290,'accent-edge')}${line(730,219,1065,290)}${line(300,354,195,440)}${line(370,354,435,440)}${line(680,354,675,440,'accent-edge')}<path d="M850 440 L1000 500 M1000 440 L850 500" stroke="${colors.accent}" stroke-width="5"/>${line(1065,354,1165,440)}<text class="small" x="600" y="555">高价值路径</text>`);
  if (kind === 'loop') return frame('Agent Loop', `${box(580,150,240,70,'Observe','cool')}${box(930,270,240,70,'Think / Plan')}${box(580,430,240,70,'Act','hot')}${box(230,270,240,70,'Evaluate','good')}${arrow(820,185,930,305)}${arrow(1050,340,820,465)}${arrow(580,465,470,305)}${arrow(350,270,580,185,'accent-edge')}<circle cx="700" cy="320" r="80" fill="none" stroke="${colors.line}" stroke-width="2"/><text class="label" x="700" y="312" text-anchor="middle">Agent</text><text class="small" x="700" y="342" text-anchor="middle">State + Goal</text>`);
  if (kind === 'timeline') return frame('异步 Runtime 时间线', `<text class="label" x="80" y="205">Model</text><text class="label" x="80" y="315">Tool</text><text class="label" x="80" y="425">Agent</text><path class="grid" d="M220 180 H1310 M220 290 H1310 M220 400 H1310"/><rect x="250" y="170" width="280" height="55" rx="10" fill="${colors.blue}"/><rect x="760" y="170" width="330" height="55" rx="10" fill="${colors.blue}"/><rect x="480" y="280" width="390" height="55" rx="10" fill="${colors.accent}"/><rect x="340" y="390" width="260" height="55" rx="10" fill="${colors.green}"/><rect x="900" y="390" width="300" height="55" rx="10" fill="${colors.green}"/><text class="small" x="250" y="500">时间 →　等待期间调度其他就绪任务，缩短关键路径</text>`);
  if (kind === 'trace') return frame('一次 Agent 请求的分布式 Trace', `<text class="label" x="80" y="190">Agent</text><text class="label" x="80" y="280">Model</text><text class="label" x="80" y="370">Tool</text><text class="label" x="80" y="460">Store</text><rect x="240" y="150" width="1030" height="48" rx="10" fill="${colors.accent}" opacity=".32"/><rect x="330" y="240" width="310" height="48" rx="10" fill="${colors.blue}" opacity=".72"/><rect x="690" y="330" width="340" height="48" rx="10" fill="${colors.green}" opacity=".72"/><rect x="1060" y="420" width="150" height="48" rx="10" fill="${colors.gold}" opacity=".8"/><text class="small" x="255" y="181">root span · agent.run</text><text class="small" x="345" y="271">model.generate</text><text class="small" x="705" y="361">tool.execute</text><text class="small" x="1070" y="451">state.save</text><path class="grid" d="M240 510 H1270"/><text class="small" x="1130" y="548">Latency →</text>`);
  if (kind === 'security') return frame('纵深防御与 Trust Boundary', `<rect x="100" y="155" width="1200" height="390" rx="34" fill="none" stroke="${colors.accent}" stroke-width="4"/><rect x="250" y="220" width="900" height="270" rx="30" fill="none" stroke="${colors.blue}" stroke-width="4"/><rect x="430" y="285" width="540" height="140" rx="26" fill="none" stroke="${colors.green}" stroke-width="4"/>${box(540,320,320,70,'Sandboxed Execution','good')}<text class="label" x="125" y="195">Identity + Authentication</text><text class="label" x="275" y="260">Authorization + Policy</text><text class="label" x="455" y="315">Capability Boundary</text><text class="small" x="1040" y="520">外部输入默认不可信</text>`);
  if (kind === 'network') return frame('多智能体协作拓扑', `${box(570,150,260,75,'Orchestrator','hot')}${box(120,340,210,70,'Research Agent','cool')}${box(420,420,210,70,'Code Agent','cool')}${box(770,420,210,70,'Review Agent','cool')}${box(1070,340,210,70,'Tool Agent','cool')}${box(570,300,260,70,'Shared Memory','good')}${arrow(620,225,270,340)}${arrow(660,225,525,420)}${arrow(740,225,875,420)}${arrow(780,225,1175,340)}${arrow(225,340,570,335)}${arrow(525,420,620,370)}${arrow(875,420,780,370)}${arrow(1175,340,830,335)}<text class="small" x="570" y="555">任务委派、共享上下文、结果汇总与共识检查</text>`);
  if (kind === 'fusion') return frame('多传感器融合', `${box(70,170,180,64,'Camera','cool')}${box(70,275,180,64,'LiDAR','cool')}${box(70,380,180,64,'IMU','cool')}${box(370,275,210,72,'Time Sync')}${box(700,275,220,72,'Sensor Fusion','hot')}${box(1060,275,250,72,'World State','good')}${arrow(250,202,370,300)}${arrow(250,307,370,311)}${arrow(250,412,370,322)}${arrow(580,311,700,311)}${arrow(920,311,1060,311,'accent-edge')}<text class="small" x="355" y="400">时间戳与坐标系对齐</text><text class="small" x="705" y="400">不确定性加权</text><text class="small" x="1080" y="400">位置、速度、对象</text>`);
  return frame('mini-llm', '');
}

await mkdir(outDir, { recursive: true });
for (const [file, match, imageName, caption, kind] of specs) {
  await writeFile(path.join(outDir, imageName), render(kind), 'utf8');
  const filePath = path.join(docsDir, file);
  const raw = await readFile(filePath, 'utf8');
  if (raw.includes(`](assets/diagrams/${imageName})`)) continue;
  const eol = raw.includes('\r\n') ? '\r\n' : '\n';
  const lines = raw.split(/\r?\n/);
  const index = lines.findIndex((line) => /^#{1,3}\s/.test(line) && line.includes(match));
  if (index < 0) throw new Error(`Cannot find heading containing "${match}" in ${file}`);
  lines.splice(index + 1, 0, '', `![${caption}](assets/diagrams/${imageName})`, '', `_图：${caption}_`);
  await writeFile(filePath, lines.join(eol), 'utf8');
}
console.log(`Generated and linked ${specs.length} mechanism diagrams.`);
