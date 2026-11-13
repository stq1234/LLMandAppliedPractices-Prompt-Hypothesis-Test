"""固定合成规则家族与确定性数据集生成器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


# Each family is held wholly within one split. Its three assignments are
# deliberately contrastive and are not new surface forms of another family's
# policy. ASTs are gold-only and are never inserted into model prompts.
FAMILIES: list[dict[str, Any]] = [
    {
        "id": "D01", "split": "dev", "complexity": "simple",
        "scenario": "项目现场访问资格核验",
        "facts": {"confidentiality_cert": "访问人员持有保密资质", "safety_training": "访问人员已完成现场安全培训", "annual_review": "访问人员已通过年度复核"},
        "rules": "项目现场访问的完整许可条件是以下三项中至少具备两项：保密资质、现场安全培训、年度复核通过；不要求三项全部具备。",
        "prohibitions": "本制度没有其他禁止条款。",
        "allow": {"op": "any", "args": [
            {"op": "all", "args": [{"op": "var", "name": "confidentiality_cert"}, {"op": "var", "name": "safety_training"}]},
            {"op": "all", "args": [{"op": "var", "name": "confidentiality_cert"}, {"op": "var", "name": "annual_review"}]},
            {"op": "all", "args": [{"op": "var", "name": "safety_training"}, {"op": "var", "name": "annual_review"}]},
        ]},
        "deny": {"op": "const", "value": False},
        "cases": [
            {"confidentiality_cert": True, "safety_training": True, "annual_review": False},
            {"confidentiality_cert": True, "safety_training": False, "annual_review": True},
            {"confidentiality_cert": True, "safety_training": False, "annual_review": False},
        ],
    },
    {
        "id": "D02", "split": "dev", "complexity": "simple",
        "scenario": "夜间运维工单放行",
        "facts": {"duty_permit": "值班运维许可有效", "mfa_ready": "双因素认证已接入", "escort_badge": "陪同人员持有夜间陪同证", "dispatch_ticket": "调度中心已登记夜间工单"},
        "rules": "夜间工单可按以下任一完整路径放行：值班运维许可有效且已接入双因素认证；或者陪同人员持夜间陪同证且调度中心已登记夜间工单。单独满足一条路径中的一项不够。",
        "prohibitions": "本流程没有其他禁止条款。",
        "allow": {"op": "any", "args": [
            {"op": "all", "args": [{"op": "var", "name": "duty_permit"}, {"op": "var", "name": "mfa_ready"}]},
            {"op": "all", "args": [{"op": "var", "name": "escort_badge"}, {"op": "var", "name": "dispatch_ticket"}]},
        ]},
        "deny": {"op": "const", "value": False},
        "cases": [
            {"duty_permit": True, "mfa_ready": True, "escort_badge": False, "dispatch_ticket": False},
            {"duty_permit": False, "mfa_ready": False, "escort_badge": True, "dispatch_ticket": True},
            {"duty_permit": True, "mfa_ready": False, "escort_badge": True, "dispatch_ticket": False},
        ],
    },
    {
        "id": "D03", "split": "dev", "complexity": "complex",
        "scenario": "档案扫描",
        "facts": {"authorized": "扫描人员已获档案管理员授权", "sealed": "档案当前处于封存状态", "emergency_order": "紧急扫描工单已登记", "supervisor_verified": "值班主管已核验紧急工单", "safety_hold_issued": "安全暂停令曾针对该档案签发", "hold_lifted": "主管已正式解除该安全暂停"},
        "rules": "档案管理员授权是所有扫描路径的必要条件。档案未封存且人员获授权时，可按常规路径扫描；档案已封存时，获授权、紧急工单已登记、值班主管已核验该工单三项同时成立，即构成完整的例外许可。除此以外没有其他正向许可条件。",
        "prohibitions": "若曾针对该档案签发安全暂停令，扫描须停止。紧急工单已登记、值班主管已核验同一工单且暂停已正式解除，这三项同时成立即解除该禁止；否则禁止仍有效。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "authorized"}, {"op": "any", "args": [{"op": "not", "arg": {"op": "var", "name": "sealed"}}, {"op": "all", "args": [{"op": "var", "name": "emergency_order"}, {"op": "var", "name": "supervisor_verified"}]}]}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "safety_hold_issued"}, {"op": "not", "arg": {"op": "all", "args": [{"op": "var", "name": "emergency_order"}, {"op": "var", "name": "supervisor_verified"}, {"op": "var", "name": "hold_lifted"}]}}]},
        "cases": [
            {"authorized": True, "sealed": False, "emergency_order": False, "supervisor_verified": False, "safety_hold_issued": False, "hold_lifted": False},
            {"authorized": True, "sealed": True, "emergency_order": True, "supervisor_verified": True, "safety_hold_issued": True, "hold_lifted": True},
            {"authorized": False, "sealed": True, "emergency_order": True, "supervisor_verified": True, "safety_hold_issued": True, "hold_lifted": False},
        ],
    },
    {
        "id": "D04", "split": "dev", "complexity": "complex",
        "scenario": "文件对外发布",
        "facts": {"owner_approved": "文件所有者已批准发布", "review_complete": "合规复核已经完成", "redaction_pending": "敏感信息脱敏仍未完成", "emergency_release": "紧急发布通道已获启动许可", "security_signed": "安全负责人已签署紧急发布意见", "court_hold_active": "法院保全令当前仍对该文件生效"},
        "rules": "完整许可有两条互相独立的路径：常规路径须同时获得文件所有者批准、完成合规复核并完成敏感信息脱敏；紧急路径须同时获得紧急发布通道许可和安全负责人签署意见，该路径可替代常规路径的三个条件。",
        "prohibitions": "法院保全令对该文件生效期间，两条发布路径都必须停止。",
        "allow": {"op": "any", "args": [
            {"op": "all", "args": [{"op": "var", "name": "owner_approved"}, {"op": "var", "name": "review_complete"}, {"op": "not", "arg": {"op": "var", "name": "redaction_pending"}}]},
            {"op": "all", "args": [{"op": "var", "name": "emergency_release"}, {"op": "var", "name": "security_signed"}]},
        ]},
        "deny": {"op": "var", "name": "court_hold_active"},
        "cases": [
            {"owner_approved": True, "review_complete": True, "redaction_pending": False, "emergency_release": False, "security_signed": False, "court_hold_active": False},
            {"owner_approved": False, "review_complete": False, "redaction_pending": True, "emergency_release": True, "security_signed": True, "court_hold_active": False},
            {"owner_approved": True, "review_complete": True, "redaction_pending": False, "emergency_release": False, "security_signed": True, "court_hold_active": True},
        ],
    },
    {
        "id": "S01", "split": "test", "complexity": "simple",
        "scenario": "自助柜取件",
        "facts": {"parcel_matched": "包裹编号与领取码相符", "recipient_verified": "收件人身份已核验"},
        "rules": "自助柜仅在包裹编号与领取码相符、且收件人身份已核验时开放取件；这两项同时满足即可以取件。",
        "prohibitions": "本流程没有额外禁止条款。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "parcel_matched"}, {"op": "var", "name": "recipient_verified"}]},
        "deny": {"op": "const", "value": False},
        "cases": [
            {"parcel_matched": True, "recipient_verified": True},
            {"parcel_matched": False, "recipient_verified": True},
            {"parcel_matched": True, "recipient_verified": False},
        ],
    },
    {
        "id": "S02", "split": "test", "complexity": "simple",
        "scenario": "园区车辆通行",
        "facts": {"bridge_unblocked": "东桥目前没有施工封闭", "ferry_operational": "渡口航线目前没有故障停航", "storm_warning": "强风预警仍在生效"},
        "rules": "车辆走东桥或渡口任一路线即可；两条路线不需要同时具备通行条件。",
        "prohibitions": "强风预警生效期间，两条路线都不得通行。",
        "allow": {"op": "any", "args": [{"op": "var", "name": "bridge_unblocked"}, {"op": "var", "name": "ferry_operational"}]},
        "deny": {"op": "var", "name": "storm_warning"},
        "cases": [
            {"bridge_unblocked": True, "ferry_operational": False, "storm_warning": False},
            {"bridge_unblocked": False, "ferry_operational": True, "storm_warning": False},
            {"bridge_unblocked": True, "ferry_operational": False, "storm_warning": True},
        ],
    },
    {
        "id": "S03", "split": "test", "complexity": "simple",
        "scenario": "执业证件备案",
        "facts": {"license_expired": "执业许可证已经过期", "insurance_registered": "责任保险已备案"},
        "rules": "备案可通过的完整条件是责任保险已备案，并且执业许可证尚未过期。",
        "prohibitions": "本流程没有额外禁止条款。",
        "allow": {"op": "all", "args": [{"op": "not", "arg": {"op": "var", "name": "license_expired"}}, {"op": "var", "name": "insurance_registered"}]},
        "deny": {"op": "const", "value": False},
        "cases": [
            {"license_expired": False, "insurance_registered": True},
            {"license_expired": True, "insurance_registered": True},
            {"license_expired": False, "insurance_registered": False},
        ],
    },
    {
        "id": "S04", "split": "test", "complexity": "simple",
        "scenario": "仓库样品出库",
        "facts": {"release_code_valid": "样品出库码有效", "seal_broken": "防拆封签已破损"},
        "rules": "样品出库码有效即满足完整的正向许可条件；除此之外没有其他正向条件。",
        "prohibitions": "防拆封签已破损时，即使出库码有效也不得出库。",
        "allow": {"op": "var", "name": "release_code_valid"},
        "deny": {"op": "var", "name": "seal_broken"},
        "cases": [
            {"release_code_valid": True, "seal_broken": False},
            {"release_code_valid": False, "seal_broken": False},
            {"release_code_valid": True, "seal_broken": True},
        ],
    },
    {
        "id": "C01", "split": "test", "complexity": "complex",
        "scenario": "企业差旅报销",
        "facts": {"account_active": "员工报销账户处于有效状态", "routine_trip": "该支出属于已登记差旅", "manager_exception": "部门经理已批准例外报销", "policy_lock": "本月费用策略锁定", "chargeback": "该笔交易已被发卡方撤销"},
        "rules": "有效账户是报销的必要条件。除此之外，完整许可还要求已登记差旅或部门经理批准例外报销至少一项成立，并且本月费用策略未锁定。",
        "prohibitions": "交易已被发卡方撤销时必须拒付。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "account_active"}, {"op": "any", "args": [{"op": "var", "name": "routine_trip"}, {"op": "var", "name": "manager_exception"}]}, {"op": "not", "arg": {"op": "var", "name": "policy_lock"}}]},
        "deny": {"op": "var", "name": "chargeback"},
        "cases": [
            {"account_active": True, "routine_trip": True, "manager_exception": False, "policy_lock": False, "chargeback": False},
            {"account_active": True, "routine_trip": False, "manager_exception": True, "policy_lock": False, "chargeback": False},
            {"account_active": True, "routine_trip": True, "manager_exception": False, "policy_lock": False, "chargeback": True},
        ],
    },
    {
        "id": "C02", "split": "test", "complexity": "complex",
        "scenario": "日间诊疗入院",
        "facts": {"consent_valid": "患者的有效知情同意书已签署", "triage_passed": "分诊检查已通过", "physician_order": "接诊医师已开具入院医嘱", "isolation_order": "隔离收治令正在生效"},
        "rules": "有效知情同意书是必要条件，但单独有同意书并不足以入院；还必须同时通过分诊检查并取得接诊医师的入院医嘱。三项共同构成完整许可条件。",
        "prohibitions": "隔离收治令生效时不得按普通日间诊疗流程入院。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "consent_valid"}, {"op": "var", "name": "triage_passed"}, {"op": "var", "name": "physician_order"}]},
        "deny": {"op": "var", "name": "isolation_order"},
        "cases": [
            {"consent_valid": True, "triage_passed": True, "physician_order": True, "isolation_order": False},
            {"consent_valid": True, "triage_passed": False, "physician_order": True, "isolation_order": False},
            {"consent_valid": True, "triage_passed": True, "physician_order": True, "isolation_order": True},
        ],
    },
    {
        "id": "C03", "split": "test", "complexity": "complex",
        "scenario": "生产线设备检修",
        "facts": {"technician_trained": "检修人员已完成设备培训", "work_permit": "有效检修许可证已签发", "incident_reported": "相关安全事件已登记", "supervisor_closed": "安全主管已正式关闭该事件"},
        "rules": "完整许可条件是检修人员完成设备培训并持有有效检修许可证；两项都满足即可进入生产线。",
        "prohibitions": "存在已登记且尚未结案的安全事件时禁止进入；安全主管正式关闭该事件后，这项禁止即解除。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "technician_trained"}, {"op": "var", "name": "work_permit"}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "incident_reported"}, {"op": "not", "arg": {"op": "var", "name": "supervisor_closed"}}]},
        "cases": [
            {"technician_trained": True, "work_permit": True, "incident_reported": False, "supervisor_closed": False},
            {"technician_trained": True, "work_permit": True, "incident_reported": True, "supervisor_closed": False},
            {"technician_trained": True, "work_permit": True, "incident_reported": True, "supervisor_closed": True},
        ],
    },
    {
        "id": "C04", "split": "test", "complexity": "complex",
        "scenario": "城市配送站暂停期装车",
        "facts": {"task_open": "配送任务仍处于开放状态", "vehicle_ready": "配送车辆检查合格", "station_paused": "配送站处于暂停期", "emergency_release": "紧急装车解除单已签发", "supervisor_verified": "值班主管已核验解除单"},
        "rules": "配送任务开放且车辆检查合格，是完整的装车许可条件；两项均满足即可装车。",
        "prohibitions": "配送站暂停期间禁止装车。已签发紧急装车解除单且值班主管已核验该单，两项同时成立即解除暂停期禁令；否则禁令继续生效。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "task_open"}, {"op": "var", "name": "vehicle_ready"}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "station_paused"}, {"op": "not", "arg": {"op": "all", "args": [{"op": "var", "name": "emergency_release"}, {"op": "var", "name": "supervisor_verified"}]}}]},
        "cases": [
            {"task_open": True, "vehicle_ready": True, "station_paused": False, "emergency_release": False, "supervisor_verified": False},
            {"task_open": True, "vehicle_ready": True, "station_paused": True, "emergency_release": True, "supervisor_verified": True},
            {"task_open": True, "vehicle_ready": False, "station_paused": True, "emergency_release": False, "supervisor_verified": False},
        ],
    },
    {
        "id": "C05", "split": "test", "complexity": "complex",
        "scenario": "供应商临时冻结解除",
        "facts": {"contract_current": "供应合同仍在有效期内", "allocation_exists": "本批货物已有采购配额", "vendor_hold": "供应商账户被临时冻结", "notice_filed": "冻结前的书面通知已备案", "urgent_order": "存在紧急采购单", "manager_approved": "采购经理已批准紧急采购"},
        "rules": "供应商合同有效且本批货物有采购配额时，货物可正常采购。",
        "prohibitions": "账户冻结期间不得采购；但冻结前已备案书面通知，或同时存在紧急采购单且采购经理已批准，任一例外成立即可解除该禁止。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "contract_current"}, {"op": "var", "name": "allocation_exists"}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "vendor_hold"}, {"op": "not", "arg": {"op": "any", "args": [{"op": "var", "name": "notice_filed"}, {"op": "all", "args": [{"op": "var", "name": "urgent_order"}, {"op": "var", "name": "manager_approved"}]}]}}]},
        "cases": [
            {"contract_current": True, "allocation_exists": True, "vendor_hold": True, "notice_filed": True, "urgent_order": False, "manager_approved": False},
            {"contract_current": True, "allocation_exists": True, "vendor_hold": False, "notice_filed": False, "urgent_order": False, "manager_approved": False},
            {"contract_current": True, "allocation_exists": True, "vendor_hold": True, "notice_filed": False, "urgent_order": True, "manager_approved": False},
        ],
    },
    {
        "id": "C06", "split": "test", "complexity": "complex",
        "scenario": "高风险岗位排班",
        "facts": {"base_qualified": "员工已通过岗位基础资格审核", "hazardous_task": "当班任务属于高风险作业", "specialist_assigned": "具备高风险资质的专员已排入本班"},
        "rules": "完整排班许可要求员工通过岗位基础资格审核，并且当班任务为非高风险作业或有高风险资质专员在本班。非高风险作业不需要该专员。",
        "prohibitions": "本排班规则没有额外禁止条款。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "base_qualified"}, {"op": "any", "args": [{"op": "not", "arg": {"op": "var", "name": "hazardous_task"}}, {"op": "var", "name": "specialist_assigned"}]}]},
        "deny": {"op": "const", "value": False},
        "cases": [
            {"base_qualified": True, "hazardous_task": False, "specialist_assigned": False},
            {"base_qualified": True, "hazardous_task": True, "specialist_assigned": True},
            {"base_qualified": True, "hazardous_task": True, "specialist_assigned": False},
        ],
    },
    {
        "id": "C07", "split": "test", "complexity": "complex",
        "scenario": "研究项目处理敏感数据",
        "facts": {"project_open": "研究项目当前仍开放", "privacy_hold_issued": "隐私审查已下达数据处理暂停", "counsel_clear": "法务已书面解除隐私暂停", "court_order": "法院保全令正在生效"},
        "rules": "完整许可条件是项目开放，并且满足以下两项之一：未曾下达隐私暂停，或法务已书面解除暂停。",
        "prohibitions": "法院保全令生效时，无论其他条件如何都不得处理。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "project_open"}, {"op": "any", "args": [{"op": "not", "arg": {"op": "var", "name": "privacy_hold_issued"}}, {"op": "var", "name": "counsel_clear"}]}]},
        "deny": {"op": "var", "name": "court_order"},
        "cases": [
            {"project_open": True, "privacy_hold_issued": False, "counsel_clear": False, "court_order": False},
            {"project_open": True, "privacy_hold_issued": True, "counsel_clear": True, "court_order": False},
            {"project_open": True, "privacy_hold_issued": True, "counsel_clear": True, "court_order": True},
        ],
    },
    {
        "id": "C08", "split": "test", "complexity": "complex",
        "scenario": "无人机巡检起飞",
        "facts": {"flight_permit": "本次飞行许可有效", "maintenance_done": "起飞前维护已完成", "routine_clearance": "常规航线已批准", "emergency_clearance": "应急航线已批准", "weather_hazard": "现场存在强风危险", "safety_override": "安全负责人已签署强风例外"},
        "rules": "完整起飞许可要求飞行许可有效、起飞前维护完成，并且常规航线或应急航线至少一条已获批准；除此之外没有其他正向条件。",
        "prohibitions": "现场存在强风危险时禁止起飞；安全负责人签署强风例外即解除这项禁止，否则禁止继续生效。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "flight_permit"}, {"op": "var", "name": "maintenance_done"}, {"op": "any", "args": [{"op": "var", "name": "routine_clearance"}, {"op": "var", "name": "emergency_clearance"}]}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "weather_hazard"}, {"op": "not", "arg": {"op": "var", "name": "safety_override"}}]},
        "cases": [
            {"flight_permit": True, "maintenance_done": True, "routine_clearance": True, "emergency_clearance": False, "weather_hazard": False, "safety_override": False},
            {"flight_permit": True, "maintenance_done": True, "routine_clearance": False, "emergency_clearance": True, "weather_hazard": True, "safety_override": True},
            {"flight_permit": True, "maintenance_done": True, "routine_clearance": True, "emergency_clearance": False, "weather_hazard": True, "safety_override": False},
        ],
    },
    {
        "id": "C09", "split": "test", "complexity": "complex",
        "scenario": "高压值守双人配置",
        "facts": {"trainee_ready": "当班人员已完成值守培训", "operator_a_certified": "甲值班员持有高压资质", "operator_b_certified": "乙值班员持有高压资质"},
        "rules": "完整配置条件是当班人员完成值守培训，并且甲、乙两名值班员中恰有一人持有高压资质；两人都持有或都没有资质都不符合配置。",
        "prohibitions": "本岗位规则没有额外禁止条款。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "trainee_ready"}, {"op": "any", "args": [{"op": "all", "args": [{"op": "var", "name": "operator_a_certified"}, {"op": "not", "arg": {"op": "var", "name": "operator_b_certified"}}]}, {"op": "all", "args": [{"op": "not", "arg": {"op": "var", "name": "operator_a_certified"}}, {"op": "var", "name": "operator_b_certified"}]}]}]},
        "deny": {"op": "const", "value": False},
        "cases": [
            {"trainee_ready": True, "operator_a_certified": True, "operator_b_certified": False},
            {"trainee_ready": True, "operator_a_certified": True, "operator_b_certified": True},
            {"trainee_ready": True, "operator_a_certified": False, "operator_b_certified": False},
        ],
    },
    {
        "id": "C10", "split": "test", "complexity": "complex",
        "scenario": "馆藏资料外借",
        "facts": {"member_active": "读者借阅资格有效", "rare_item": "该资料属于珍贵馆藏", "librarian_approval": "馆员已书面批准珍藏资料外借", "age_restriction": "读者未达到普通借阅年龄", "guardian_consent": "监护人已提供书面同意"},
        "rules": "完整许可要求读者借阅资格有效；若资料属于珍贵馆藏，还必须取得馆员书面批准，普通资料则不需要这项批准。除此以外没有其他正向条件。",
        "prohibitions": "未达到普通借阅年龄时不得借阅；监护人书面同意即解除这项年龄限制。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "member_active"}, {"op": "any", "args": [{"op": "not", "arg": {"op": "var", "name": "rare_item"}}, {"op": "var", "name": "librarian_approval"}]}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "age_restriction"}, {"op": "not", "arg": {"op": "var", "name": "guardian_consent"}}]},
        "cases": [
            {"member_active": True, "rare_item": False, "librarian_approval": False, "age_restriction": False, "guardian_consent": False},
            {"member_active": True, "rare_item": True, "librarian_approval": False, "age_restriction": False, "guardian_consent": False},
            {"member_active": True, "rare_item": True, "librarian_approval": True, "age_restriction": True, "guardian_consent": True},
        ],
    },
    {
        "id": "C11", "split": "test", "complexity": "complex",
        "scenario": "洪水闸门检修通行",
        "facts": {"work_authorized": "检修任务已获授权", "lockout_done": "设备断电上锁已完成", "restricted_zone": "检修位置属于限制区域", "rescue_team": "应急救援队已到场", "command_clearance": "现场指挥已批准进入", "facility_shutdown": "闸站全站停运令正在生效", "emergency_team": "防汛应急小组已到场", "controller_clearance": "闸站控制员已签发停运期通行许可"},
        "rules": "进入任何区域都必须先取得检修任务授权并完成设备断电上锁。进入限制区域还必须同时满足应急救援队到场和现场指挥批准；非限制区域不需要这两项附加条件。",
        "prohibitions": "全站停运期间不得进入。防汛应急小组到场且闸站控制员签发停运期通行许可，两项同时成立即解除这项禁止；否则禁止继续生效。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "work_authorized"}, {"op": "var", "name": "lockout_done"}, {"op": "any", "args": [{"op": "not", "arg": {"op": "var", "name": "restricted_zone"}}, {"op": "all", "args": [{"op": "var", "name": "rescue_team"}, {"op": "var", "name": "command_clearance"}]}]}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "facility_shutdown"}, {"op": "not", "arg": {"op": "all", "args": [{"op": "var", "name": "emergency_team"}, {"op": "var", "name": "controller_clearance"}]}}]},
        "cases": [
            {"work_authorized": True, "lockout_done": True, "restricted_zone": False, "rescue_team": False, "command_clearance": False, "facility_shutdown": False, "emergency_team": False, "controller_clearance": False},
            {"work_authorized": True, "lockout_done": True, "restricted_zone": True, "rescue_team": True, "command_clearance": True, "facility_shutdown": False, "emergency_team": False, "controller_clearance": False},
            {"work_authorized": True, "lockout_done": True, "restricted_zone": False, "rescue_team": False, "command_clearance": False, "facility_shutdown": True, "emergency_team": False, "controller_clearance": False},
        ],
    },
    {
        "id": "C12", "split": "test", "complexity": "complex",
        "scenario": "研究数据跨境导出",
        "facts": {"request_logged": "数据导出请求已登记", "privacy_screen_passed": "隐私审查已通过", "overseas_transfer": "本次导出目的地在境外", "dpa_approved": "数据处理协议已获批准", "injunction": "数据保全禁令正在生效", "legal_release": "法院已书面解除数据保全禁令"},
        "rules": "完整许可要求请求已登记且隐私审查通过；目的地在境内时即可导出，目的地在境外时还必须先获批数据处理协议。除此以外没有其他正向条件。",
        "prohibitions": "数据保全禁令生效时不得导出；法院书面解除禁令即解除这项禁止，否则禁止仍有效。",
        "allow": {"op": "all", "args": [{"op": "var", "name": "request_logged"}, {"op": "var", "name": "privacy_screen_passed"}, {"op": "any", "args": [{"op": "not", "arg": {"op": "var", "name": "overseas_transfer"}}, {"op": "var", "name": "dpa_approved"}]}]},
        "deny": {"op": "all", "args": [{"op": "var", "name": "injunction"}, {"op": "not", "arg": {"op": "var", "name": "legal_release"}}]},
        "cases": [
            {"request_logged": True, "privacy_screen_passed": True, "overseas_transfer": False, "dpa_approved": False, "injunction": False, "legal_release": False},
            {"request_logged": True, "privacy_screen_passed": True, "overseas_transfer": True, "dpa_approved": True, "injunction": False, "legal_release": False},
            {"request_logged": True, "privacy_screen_passed": True, "overseas_transfer": True, "dpa_approved": False, "injunction": False, "legal_release": False},
        ],
    },
]


def eval_ast(expr: dict[str, Any], facts: dict[str, bool]) -> bool:
    op = expr["op"]
    if op == "const":
        return bool(expr["value"])
    if op == "var":
        return bool(facts[expr["name"]])
    if op == "not":
        return not eval_ast(expr["arg"], facts)
    if op == "all":
        return all(eval_ast(arg, facts) for arg in expr["args"])
    if op == "any":
        return any(eval_ast(arg, facts) for arg in expr["args"])
    raise ValueError(f"未知 AST 运算符：{op}")


def make_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        if len(family["cases"]) != 3:
            raise ValueError(f"{family['id']} 必须恰有三个赋值")
        known = set(family["facts"])
        for variant, case_facts in enumerate(family["cases"], 1):
            if set(case_facts) != known:
                raise ValueError(f"{family['id']} 样本 {variant} 的事实键不完整")
            allow_value = eval_ast(family["allow"], case_facts)
            deny_value = eval_ast(family["deny"], case_facts)
            decision = "DENY" if deny_value or not allow_value else "ALLOW"
            positives = [label for key, label in family["facts"].items() if case_facts[key]]
            negatives = [label for key, label in family["facts"].items() if not case_facts[key]]
            if deny_value:
                path = "命中禁止条款，禁止优先"
                rationale = "禁止条件为真，因此拒绝；许可条件是否满足不改变该结论。"
            elif allow_value:
                path = "许可组合成立且无禁止命中"
                rationale = "许可组合为真且没有生效的禁止条款，因此允许。"
            else:
                path = "没有许可组合成立且无禁止命中"
                rationale = "没有完整许可组合成立，因此拒绝。"
            records.append({
                "sample_id": f"{family['id']}-V{variant}",
                "family_id": family["id"],
                "split": family["split"],
                "complexity": family["complexity"],
                "scenario": family["scenario"],
                "variant": variant,
                "rules_text": family["rules"],
                "prohibitions_text": family["prohibitions"],
                "facts": case_facts,
                "fact_labels": family["facts"],
                "facts_text": "当前事实：" + "；".join(f"{label}：{'是' if case_facts[key] else '否'}" for key, label in family["facts"].items()) + "。",
                "gold_ast": {"allow": family["allow"], "deny": family["deny"]},
                "gold_eval": {"allow_value": allow_value, "deny_value": deny_value, "decision_path": path},
                "gold_decision": decision,
                "gold_rationale": rationale,
                "gold_true_facts": positives,
                "gold_false_facts": negatives,
            })
    return records


def write_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    records = make_records()
    dest = path or (DATA_DIR / "dataset.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return records


if __name__ == "__main__":
    write_dataset()
