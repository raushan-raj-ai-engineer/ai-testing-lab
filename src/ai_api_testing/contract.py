from typing import Any
from src.ai_api_testing.models import ContractIssue
TYPE_MAP={'string':str,'integer':int,'number':(int,float),'boolean':bool,'object':dict,'array':list}
def validate_contract(payload:dict[str,Any],schema:dict[str,Any],*,path='$'):
    issues=[]; required=schema.get('required',[]); props=schema.get('properties',{})
    for k in required:
        if k not in payload: issues.append(ContractIssue(f'{path}.{k}','required field missing'))
    for k,v in payload.items():
        spec=props.get(k)
        if spec is None:
            if schema.get('additionalProperties',True) is False: issues.append(ContractIssue(f'{path}.{k}','unexpected field'))
            continue
        expected=spec.get('type'); py=TYPE_MAP.get(expected)
        if py is not None and not isinstance(v,py): issues.append(ContractIssue(f'{path}.{k}',f'expected {expected}, got {type(v).__name__}'))
        if expected=='object' and isinstance(v,dict): issues.extend(validate_contract(v,spec,path=f'{path}.{k}'))
    return issues
